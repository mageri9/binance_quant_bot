import pandas as pd
from loguru import logger

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

        # Реальная отправка ордера на биржу
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                side=side,
                order_type="market",
                amount=adjusted_amount,
                price=latest_close,
            )

            pnl_str = (
                f", PnL: {order['pnl']:.2f}$" if order.get("pnl") is not None else ""
            )
            msg = (
                f"🚀 [ORDER {order['status'].upper()}] Исполнен ордер {side.upper()} по {symbol}. "
                f"Цена: {order['price']:.2f}$, Количество: {order['amount']:.6f}{pnl_str}."
            )
            logger.info(msg)
            return msg
        except Exception as e:
            err_msg = (
                f"🚨 [API ERROR] Ошибка отправки ордера {side.upper()} по {symbol}: {e}"
            )
            logger.error(err_msg)

            # Аварийная остановка при сбое API
            await self.kill_switch_manager.set_state(
                state=KillSwitchState.SAFE_MODE,
                reason="API_FAILURE",
                details=str(e),
            )
            return err_msg