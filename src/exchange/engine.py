import pandas as pd
from loguru import logger

from datetime import datetime, timezone
from src.exchange.base import BaseExchange
from src.risk.engine import RiskEngine, RiskDecision
from src.risk.kill_switch import KillSwitchManager, KillSwitchState
from src.crud.paper import PaperTradingRepository


class TradingEngine:
    """
    Универсальный торговый движок, связывающий Биржу, Риски и Kill Switch.
    """

    def __init__(
        self,
        exchange: BaseExchange,
        risk_engine: RiskEngine,
        kill_switch_manager: KillSwitchManager,
        session,
        settings,
    ):
        self.exchange = exchange
        self.risk_engine = risk_engine
        self.kill_switch_manager = kill_switch_manager
        self.session = session
        self.settings = settings
        self.repo = PaperTradingRepository(session)

    async def process_signal(
        self, symbol: str, signal: int, latest_close: float
    ) -> str | None:
        if await self.kill_switch_manager.is_trading_blocked():
            logger.warning(
                f"[TradingEngine] Торговля по {symbol} заблокирована: Kill Switch активен."
            )
            return None

        if signal not in [1, -1]:
            return None

        side = "buy" if signal == 1 else "sell"

        balance = await self.exchange.get_balance()
        pos = await self.exchange.get_position(symbol)
        open_positions = [pos] if pos else []

        closed_trades = await self.repo.get_closed_trades(symbol, limit=20)
        closed_trades_dicts = [{"pnl": t.pnl} for t in closed_trades]

        consecutive_losses = 0
        for t in reversed(closed_trades):
            if t.pnl is not None:
                if t.pnl < 0:
                    consecutive_losses += 1
                else:
                    break

        requested_value = balance["total"] * self.settings.PAPER_RISK_PCT
        requested_amount = requested_value / latest_close

        # Проверка рисков
        decision, adjusted_amount, reason = await self.risk_engine.validate_signal(
            symbol=symbol,
            side=side,
            requested_amount=requested_amount,
            current_price=latest_close,
            balance_free=balance["free"],
            balance_total=balance["total"],
            open_positions=open_positions,
            closed_trades_last_24h=closed_trades_dicts,
            consecutive_losses=consecutive_losses,
        )

        if decision == RiskDecision.DENY:
            msg = f"🚫 [RISK DENY] Сделка {side.upper()} по {symbol} ОТКЛОНЕНА. Причина: {reason}"
            logger.warning(msg)
            return msg

        # Режим Shadow Trading (Dry Run)
        if self.settings.SHADOW_TRADING:
            msg = (
                f"👤 [SHADOW TRADING] Одобрен ордер {side.upper()} {symbol}. "
                f"Объем: {adjusted_amount:.6f} монет по цене {latest_close:.2f}$. "
                f"Решение рисков: {decision.value} ({reason}). "
                f"Ордер НЕ отправлен на биржу."
            )
            logger.info(msg)
            return msg

        sl_pct = self.settings.PAPER_SL_PCT
        tp_pct = self.settings.PAPER_TP_PCT

        if side == "buy":
            sl_price = latest_close * (1.0 - sl_pct)
            tp_price = latest_close * (1.0 + tp_pct)
            close_side = "sell"
        else:
            sl_price = latest_close * (1.0 + sl_pct)
            tp_price = latest_close * (1.0 - tp_pct)
            close_side = "buy"

        # Реальная отправка ордера на биржу
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                side=side,
                order_type="market",
                amount=adjusted_amount,
                price=latest_close,
            )

            stop_warning = ""
            if hasattr(self.exchange, "create_stop_orders"):
                try:
                    stop_result = await self.exchange.create_stop_orders(
                        symbol=symbol,
                        side=close_side,
                        amount=order["amount"],
                        sl_price=sl_price,
                        tp_price=tp_price,
                    )
                    if not stop_result.get("sl_order_id") or not stop_result.get("tp_order_id"):
                        stop_warning = " ⚠️ SL/TP выставлены не полностью, проверьте позицию на бирже вручную!"
                except Exception as stop_err:
                    stop_warning = (
                        " ⚠️ SL/TP НЕ выставлены, проверьте позицию на бирже вручную!"
                    )
                    logger.error(
                        f"[TradingEngine] Не удалось выставить SL/TP по {symbol} после входа: {stop_err}"
                    )

            db_warning = ""
            try:
                entry_candle_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                await self.repo.create_trade(
                    symbol=symbol,
                    entry_price=order["price"],
                    amount=order["amount"],
                    sl_price=sl_price,
                    tp_price=tp_price,
                    entry_candle_time=entry_candle_time,
                    is_short=(side == "sell"),
                )
            except Exception as db_err:
                db_warning = " ⚠️ Запись в БД не удалась, сверьте позицию вручную!"
                logger.error(
                    f"[TradingEngine] Ордер {side.upper()} {symbol} исполнен на бирже, "
                    f"но не записан в БД: {db_err}"
                )

            pnl_str = (
                f", PnL: {order['pnl']:.2f}$" if order.get("pnl") is not None else ""
            )
            msg = (
                f"🚀 [ORDER {order['status'].upper()}] Исполнен ордер {side.upper()} по {symbol}. "
                f"Цена: {order['price']:.2f}$, Количество: {order['amount']:.6f}{pnl_str}."
                f"{stop_warning}, {db_warning}"
            )
            logger.info(msg)
            return msg
        except Exception as e:
            err_msg = (
                f"🚨 [API ERROR] Ошибка отправки ордера {side.upper()} по {symbol}: {e}"
            )
            logger.error(err_msg)

            await self.kill_switch_manager.set_state(
                state=KillSwitchState.SAFE_MODE,
                reason="API_FAILURE",
                details=str(e),
            )
            return err_msg


    async def check_and_close_positions(self, symbol: str) -> str | None:
        """
        Проверяет, не закрылась ли живая позиция по symbol на бирже (сработал SL/TP).
        Если да — отменяет второй "повисший" reduce-only ордер и фиксирует сделку в БД.
        """
        db_trade = await self.repo.get_active_trade(symbol)
        if db_trade is None:
            return None

        ex_pos = await self.exchange.get_position(symbol)
        if ex_pos is not None:
            return None  # позиция всё ещё открыта, всё штатно

        # Позиция на бирже закрыта, а в БД числится открытой — SL/TP сработал.
        # Безопасно отменяем все повисшие ордера по данному активу.
        if hasattr(self.exchange, "get_open_orders"):
            try:
                leftover_orders = await self.exchange.get_open_orders(symbol)
                for o in leftover_orders:
                    try:
                        await self.exchange.cancel_order(o["id"], symbol)
                    except Exception as cancel_err:
                        logger.warning(
                            f"[TradingEngine] Не удалось отменить повисший ордер {o['id']} по {symbol}: {cancel_err}"
                        )
            except Exception as get_orders_err:
                logger.error(
                    f"[TradingEngine] Не удалось получить открытые ордера для отмены по {symbol}: {get_orders_err}"
                )

        exit_price = None
        if hasattr(self.exchange, "get_last_trade_price"):
            try:
                exit_price = await self.exchange.get_last_trade_price(symbol)
            except Exception as exit_price_err:
                logger.error(
                    f"[TradingEngine] Не удалось получить реальную цену закрытия для {symbol}: {exit_price_err}"
                )

        if exit_price is None:
            exit_price = db_trade.sl_price or db_trade.tp_price or db_trade.entry_price

        if db_trade.is_short:
            pnl = (db_trade.entry_price - exit_price) * db_trade.amount
        else:
            pnl = (exit_price - db_trade.entry_price) * db_trade.amount

        # Безопасное закрытие под локом в репозитории
        await self.repo.close_trade(db_trade, exit_price, pnl)

        msg = (
            f"✅ [LIVE CLOSE] Позиция по {symbol} закрыта на бирже. "
            f"Цена выхода: {exit_price:.2f}$, PnL: {pnl:.2f}$"
        )
        logger.info(msg)
        return msg