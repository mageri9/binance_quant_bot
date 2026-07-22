import hashlib
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from loguru import logger

from src.crud.execution import ExecutionRepository
from src.crud.paper import TradeRepository
from src.exchange.base import BaseExchange
from src.risk.engine import RiskDecision, RiskEngine
from src.risk.kill_switch import KillSwitchManager, KillSwitchState
from src.telegram.formatter import TradingNotification
from src.events import EventStore


class TradingEngine:
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
        self.repo = TradeRepository(session)
        self.execution_repo = ExecutionRepository(session)

    async def process_signal(
        self,
        symbol: str,
        signal: int,
        latest_close: float,
        *,
        model_id: str | None = None,
        prediction_id: int | None = None,
        idempotency_key: str | None = None,
    ) -> str | None:
        if await self.kill_switch_manager.is_trading_blocked():
            logger.warning(f"[TradingEngine] Торговля по {symbol} заблокирована Kill Switch.")
            return None
        if signal not in (1, -1):
            return None

        side = "buy" if signal == 1 else "sell"
        mode = _get_execution_mode(self.settings)
        balance, position = await self._get_account_context(symbol, mode)

        closed_trades = await self.repo.get_closed_trades(
            symbol, limit=20, environment="paper" if mode in {"paper", "shadow"} else mode
        )
        consecutive_losses = _count_consecutive_losses(closed_trades)
        requested_value = balance["total"] * self.settings.PAPER_RISK_PCT
        requested_amount = requested_value / latest_close

        decision, adjusted_amount, reason = await self.risk_engine.validate_signal(
            symbol=symbol,
            side=side,
            requested_amount=requested_amount,
            current_price=latest_close,
            balance_free=balance["free"],
            balance_total=balance["total"],
            open_positions=[position] if position else [],
            closed_trades_last_24h=[{"pnl": float(trade.pnl) if trade.pnl is not None else None} for trade in closed_trades],
            consecutive_losses=consecutive_losses,
        )
        if decision == RiskDecision.DENY:
            message = (
                f"🚫 <b>{symbol} · RISK DENY</b>\n\n"
                f"Сигнал: <code>{side.upper()}</code>\nПричина: {reason}"
            )
            logger.warning(message)
            return message

        is_close_order = _is_close_order(position, side)
        sl_price, tp_price, close_side = _protection_prices(
            side,
            latest_close,
            self.settings.PAPER_SL_PCT,
            self.settings.PAPER_TP_PCT,
        )

        if mode == "shadow":
            return _format_shadow_decision(
                symbol, side, adjusted_amount, latest_close, decision, reason
            )
        if mode == "paper":
            return await self._process_paper_order(
                symbol=symbol,
                side=side,
                amount=adjusted_amount,
                price=latest_close,
                sl_price=sl_price,
                tp_price=tp_price,
                is_close_order=is_close_order,
                model_id=model_id,
            )

        return await self._process_live_order(
            mode=mode,
            symbol=symbol,
            side=side,
            close_side=close_side,
            amount=adjusted_amount,
            reference_price=latest_close,
            sl_price=sl_price,
            tp_price=tp_price,
            is_close_order=is_close_order,
            model_id=model_id,
            prediction_id=prediction_id,
            idempotency_key=idempotency_key,
        )

    async def _get_account_context(self, symbol: str, mode: str) -> tuple[dict, dict | None]:
        if mode in {"paper", "shadow"}:
            portfolio = await self.repo.get_portfolio()
            active = await self.repo.get_active_trade(symbol, "paper")
            position = _trade_as_position(active) if active else None
            # Risk sizing is a float-domain model; accounting remains Decimal
            # inside the repository and is converted only at this boundary.
            return {"free": float(portfolio.cash), "total": float(portfolio.balance)}, position
        return await self.exchange.get_balance(), await self.exchange.get_position(symbol)

    async def _process_live_order(
        self,
        *,
        mode: str,
        symbol: str,
        side: str,
        close_side: str,
        amount: float,
        reference_price: float,
        sl_price: float,
        tp_price: float,
        is_close_order: bool,
        model_id: str | None,
        prediction_id: int | None,
        idempotency_key: str | None,
    ) -> str:
        stable_key = idempotency_key or str(uuid.uuid4())
        client_order_id = _make_client_order_id(mode, stable_key)
        correlation_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"marketmind:{mode}:{stable_key}"))

        intent, created = await self.execution_repo.create_intent(
            correlation_id=correlation_id,
            client_order_id=client_order_id,
            environment=mode,
            symbol=symbol,
            side=side,
            order_type="market",
            requested_amount=amount,
            requested_price=reference_price,
            sl_price=None if is_close_order else sl_price,
            tp_price=None if is_close_order else tp_price,
            model_id=model_id,
            prediction_id=prediction_id,
        )
        if not created and intent.trade_id is not None:
            logger.info(f"[TradingEngine] Дубликат {client_order_id} проигнорирован.")
            return ""

        if not created and intent.status == "FILLED":
            order = _order_from_intent(intent)
            order["recovered"] = True
        else:
            if not created:
                # A previously persisted market attempt is ambiguous. Recovery
                # may look it up, but it must never submit it again.
                try:
                    order = await self.exchange.get_order_by_client_id(symbol, client_order_id)
                except Exception as exc:
                    order = None
                    recovery_error = exc
                else:
                    recovery_error = None
                if order is None:
                    await self.kill_switch_manager.set_state(
                        KillSwitchState.SAFE_MODE,
                        "ORDER_RECOVERY_REQUIRED",
                        f"{symbol} {client_order_id}: {recovery_error or 'order not found'}",
                    )
                    return f"🚨 <b>{symbol} · ORDER RECOVERY REQUIRED</b>"
            else:
                await self.execution_repo.mark_submitted(intent)
                try:
                    order = await self.exchange.create_order(
                        symbol=symbol,
                        side=side,
                        order_type="market",
                        amount=amount,
                        price=reference_price,
                        client_order_id=client_order_id,
                        reduce_only=is_close_order,
                    )
                except Exception as exc:
                    # A timeout/lost response is not a rejection. Preserve the
                    # submitted intent for lookup-only recovery.
                    await self.execution_repo.mark_submission_uncertain(intent, str(exc))
                    await self.kill_switch_manager.set_state(
                        KillSwitchState.SAFE_MODE,
                        "ORDER_SUBMISSION_UNCERTAIN",
                        f"{side.upper()} {symbol} {client_order_id}: {exc}",
                    )
                    return f"🚨 <b>{symbol} · ORDER RECOVERY REQUIRED</b>"

            try:
                await self.execution_repo.apply_exchange_order(intent, order)
                await self.execution_repo.record_fills(mode, order)
            except Exception as ledger_exc:
                await self.session.rollback()
                await self.kill_switch_manager.set_state(
                    KillSwitchState.SAFE_MODE,
                    "EXECUTION_LEDGER_WRITE_FAILED",
                    f"{symbol} {client_order_id}: {ledger_exc}",
                )
                logger.critical(
                    f"[TradingEngine] Биржа приняла {client_order_id}, но ledger не обновлён: {ledger_exc}"
                )
                return (
                    f"🚨 <b>{symbol} · LEDGER ERROR</b>\n\n"
                    "Ордер мог быть исполнен на Binance. Торговля остановлена до reconciliation.\n"
                    f"Client ID: <code>{client_order_id}</code>"
                )

        if order.get("status") not in {"filled", "closed"}:
            return (
                f"🟡 <b>{symbol} · ORDER {str(order.get('status')).upper()}</b>\n\n"
                f"Ордер: <code>{order.get('order_id') or client_order_id}</code>\n"
                "Позиция обновится только после подтверждённого fill."
            )

        fill_price = float(order.get("average_price") or order.get("price") or reference_price)
        filled_amount = float(order.get("filled_amount") or order.get("amount") or 0)
        if filled_amount <= 0:
            await self.kill_switch_manager.set_state(
                KillSwitchState.SAFE_MODE,
                "INVALID_FILL",
                f"{symbol} {client_order_id}: filled amount is zero",
            )
            return f"🚨 <b>{symbol} · INVALID FILL</b>\n\nТорговля остановлена."

        if is_close_order:
            return await self._project_live_close(
                intent, order, symbol, side, fill_price, filled_amount
            )

        sl_ok, tp_ok, stop_warning = await self._place_protection(
            mode=mode,
            stable_key=stable_key,
            parent_intent=intent,
            symbol=symbol,
            close_side=close_side,
            amount=filled_amount,
            sl_price=sl_price,
            tp_price=tp_price,
            model_id=model_id,
        )
        if not sl_ok and not tp_ok:
            return await self._emergency_close(
                mode=mode,
                stable_key=stable_key,
                intent=intent,
                entry_order=order,
                symbol=symbol,
                entry_side=side,
                close_side=close_side,
                amount=filled_amount,
                entry_price=fill_price,
                sl_price=sl_price,
                tp_price=tp_price,
                model_id=model_id,
            )

        db_warning = None
        try:
            trade = await self.repo.create_trade(
                symbol=symbol,
                entry_price=fill_price,
                amount=filled_amount,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_candle_time=int(datetime.now(timezone.utc).timestamp() * 1000),
                is_short=(side == "sell"),
                source="binance",
                environment=mode,
                client_order_id=client_order_id,
                entry_order_id=order.get("order_id"),
                model_id=model_id,
                update_portfolio=False,
            )
            await self.execution_repo.link_trade(intent, trade.id)
        except Exception as db_exc:
            await self.session.rollback()
            db_warning = "⚠️ Позиция исполнена, но локальная проекция не записана."
            await self.kill_switch_manager.set_state(
                KillSwitchState.SAFE_MODE,
                "POSITION_PROJECTION_WRITE_FAILED",
                f"{symbol} {client_order_id}: {db_exc}",
            )
            logger.critical(f"[TradingEngine] Не записана live-позиция {symbol}: {db_exc}")

        warning = " ".join(part for part in (stop_warning, db_warning) if part) or None
        event = _format_position_opened(
            symbol=symbol,
            side=side,
            price=fill_price,
            amount=filled_amount,
            protection=f"SL {'✅' if sl_ok else '⚠️'} · TP {'✅' if tp_ok else '⚠️'}",
            order_id=order.get("order_id") or client_order_id,
            model_id=model_id,
            warning=warning,
        )
        await self._publish_notification(event, correlation_id)
        return event

    async def _record_order_failure(self, intent, symbol: str, side: str, exc: Exception) -> None:
        try:
            await self.execution_repo.mark_failed(intent, str(exc))
        except Exception as ledger_exc:
            await self.session.rollback()
            logger.critical(f"[TradingEngine] Не записана ошибка intent: {ledger_exc}")
        await self.kill_switch_manager.set_state(
            KillSwitchState.SAFE_MODE,
            "API_FAILURE",
            f"{side.upper()} {symbol}: {exc}",
        )

    async def _project_live_close(
        self, intent, order: dict, symbol: str, side: str, price: float, amount: float
    ) -> str:
        trade = await self.repo.get_active_trade(symbol, _get_execution_mode(self.settings))
        # Futures realized PnL is exchange-reported. Do not derive it from
        # entry price, leverage/margin and fees are not represented locally.
        pnl = order.get("realized_pnl")
        if trade is not None:
            await self.repo.close_trade(
                trade,
                price,
                pnl,
                exit_order_id=order.get("order_id"),
                update_portfolio=False,
            )
            await self.execution_repo.link_trade(intent, trade.id)
        event = _format_position_closed(
            symbol=symbol,
            side=side,
            price=price,
            amount=amount,
            pnl=pnl,
            order_id=order.get("order_id") or intent.client_order_id,
            commissions=order.get("commission"),
        )
        await self._publish_notification(event, intent.correlation_id)
        return event

    async def _place_protection(
        self,
        *,
        mode: str,
        stable_key: str,
        parent_intent,
        symbol: str,
        close_side: str,
        amount: float,
        sl_price: float,
        tp_price: float,
        model_id: str | None,
    ) -> tuple[bool, bool, str | None]:
        if not hasattr(self.exchange, "create_stop_orders"):
            return False, False, "⚠️ Exchange adapter не поддерживает SL/TP."
        sl_client_id = _make_client_order_id(mode, f"{stable_key}:sl")
        tp_client_id = _make_client_order_id(mode, f"{stable_key}:tp")
        correlation_base = f"marketmind:{mode}:{stable_key}"
        try:
            sl_intent, _ = await self.execution_repo.create_intent(
                correlation_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{correlation_base}:sl")),
                client_order_id=sl_client_id,
                environment=mode,
                symbol=symbol,
                side=close_side,
                order_type="STOP_MARKET",
                requested_amount=amount,
                requested_price=sl_price,
                sl_price=sl_price,
                tp_price=None,
                model_id=model_id,
                purpose="STOP_LOSS",
                parent_intent_id=parent_intent.id,
                reduce_only=True,
            )
            tp_intent, _ = await self.execution_repo.create_intent(
                correlation_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{correlation_base}:tp")),
                client_order_id=tp_client_id,
                environment=mode,
                symbol=symbol,
                side=close_side,
                order_type="TAKE_PROFIT_MARKET",
                requested_amount=amount,
                requested_price=tp_price,
                sl_price=None,
                tp_price=tp_price,
                model_id=model_id,
                purpose="TAKE_PROFIT",
                parent_intent_id=parent_intent.id,
                reduce_only=True,
            )
            result = await self.exchange.create_stop_orders(
                symbol=symbol,
                side=close_side,
                amount=amount,
                sl_price=sl_price,
                tp_price=tp_price,
                sl_client_order_id=sl_client_id,
                tp_client_order_id=tp_client_id,
            )
            sl_ok = bool(result.get("sl_order_id"))
            tp_ok = bool(result.get("tp_order_id"))
            if result.get("sl_order"):
                await self.execution_repo.apply_exchange_order(sl_intent, result["sl_order"])
            elif not sl_ok:
                await self.execution_repo.mark_failed(sl_intent, "Binance did not accept stop-loss")
            if result.get("tp_order"):
                await self.execution_repo.apply_exchange_order(tp_intent, result["tp_order"])
            elif not tp_ok:
                await self.execution_repo.mark_failed(tp_intent, "Binance did not accept take-profit")
            if not sl_ok or not tp_ok:
                await self.kill_switch_manager.set_state(
                    KillSwitchState.SAFE_MODE,
                    "UNPROTECTED_POSITION",
                    f"{symbol}: SL={sl_ok}, TP={tp_ok}",
                )
            warning = None if sl_ok and tp_ok else "⚠️ Защита выставлена не полностью."
            return sl_ok, tp_ok, warning
        except Exception as exc:
            logger.error(f"[TradingEngine] Ошибка установки защиты {symbol}: {exc}")
            return False, False, "⚠️ SL/TP не выставлены."

    async def _emergency_close(
        self,
        *,
        mode: str,
        stable_key: str,
        intent,
        entry_order: dict,
        symbol: str,
        entry_side: str,
        close_side: str,
        amount: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        model_id: str | None,
    ) -> str:
        emergency_client_id = _make_client_order_id(mode, f"{stable_key}:emergency-close")
        try:
            emergency_intent, _ = await self.execution_repo.create_intent(
                correlation_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"marketmind:{mode}:{stable_key}:emergency-close",
                    )
                ),
                client_order_id=emergency_client_id,
                environment=mode,
                symbol=symbol,
                side=close_side,
                order_type="market",
                requested_amount=amount,
                requested_price=None,
                sl_price=None,
                tp_price=None,
                model_id=model_id,
                purpose="EMERGENCY_CLOSE",
                parent_intent_id=intent.id,
                reduce_only=True,
            )
            close_order = await self.exchange.create_order(
                symbol=symbol,
                side=close_side,
                order_type="market",
                amount=amount,
                client_order_id=emergency_client_id,
                reduce_only=True,
            )
            await self.execution_repo.apply_exchange_order(emergency_intent, close_order)
            await self.execution_repo.record_fills(mode, close_order)
            exit_price = float(
                close_order.get("average_price") or close_order.get("price") or entry_price
            )
            pnl = close_order.get("realized_pnl")
            try:
                trade = await self.repo.create_trade(
                    symbol=symbol,
                    entry_price=entry_price,
                    amount=amount,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    entry_candle_time=int(datetime.now(timezone.utc).timestamp() * 1000),
                    is_short=(entry_side == "sell"),
                source="binance",
                environment=mode,
                    client_order_id=intent.client_order_id,
                    entry_order_id=entry_order.get("order_id"),
                    model_id=model_id,
                    update_portfolio=False,
                )
                await self.repo.close_trade(
                    trade,
                    exit_price,
                    pnl,
                    exit_order_id=close_order.get("order_id"),
                    update_portfolio=False,
                )
                await self.execution_repo.link_trade(intent, trade.id)
                await self.execution_repo.link_trade(emergency_intent, trade.id)
            except Exception as db_exc:
                await self.session.rollback()
                logger.critical(f"[TradingEngine] Emergency close не записан: {db_exc}")
            return (
                f"🚨 <b>{symbol} · EMERGENCY CLOSE</b>\n\n"
                "Оба защитных ордера отклонены; позиция немедленно закрыта.\n"
                f"Выход: <code>{exit_price:.2f}$</code>\nPnL: <code>{pnl:+.2f}$</code>"
            )
        except Exception as exc:
            await self.kill_switch_manager.set_state(
                KillSwitchState.SAFE_MODE,
                "UNPROTECTED_POSITION",
                f"{symbol}: {exc}",
            )
            logger.critical(f"[TradingEngine] Незащищённая позиция {symbol}: {exc}")
            return (
                f"🚨 <b>{symbol} · НЕЗАЩИЩЁННАЯ ПОЗИЦИЯ</b>\n\n"
                "SL/TP и аварийное закрытие не выполнены. Требуется ручное вмешательство."
            )

    async def _process_paper_order(
        self,
        *,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        sl_price: float,
        tp_price: float,
        is_close_order: bool,
        model_id: str | None,
    ) -> str:
        amount = Decimal(str(amount))
        price = Decimal(str(price))
        sl_price = Decimal(str(sl_price))
        tp_price = Decimal(str(tp_price))
        active = await self.repo.get_active_trade(symbol, "paper")
        if is_close_order and active is not None:
            pnl = (
                (active.entry_price - price) * amount
                if active.is_short
                else (price - active.entry_price) * amount
            )
            await self.repo.close_trade(active, price, pnl)
            event = _format_position_closed(
                symbol=symbol,
                side=side,
                price=price,
                amount=amount,
                pnl=pnl,
                order_id="paper",
            )
            await self._publish_notification(event, str(uuid.uuid4()))
            return event

        trade = await self.repo.create_trade(
            symbol=symbol,
            entry_price=price,
            amount=amount,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_candle_time=int(datetime.now(timezone.utc).timestamp() * 1000),
            is_short=(side == "sell"),
            source="paper",
            environment="paper",
            model_id=model_id,
        )
        event = _format_position_opened(
            symbol=symbol,
            side=side,
            price=price,
            amount=amount,
            protection="SL ✅ · TP ✅ (симуляция)",
            order_id=f"paper-{trade.id}",
            model_id=model_id,
        )
        await self._publish_notification(event, str(uuid.uuid4()))
        return event

    async def check_and_close_positions(self, symbol: str) -> str | None:
        """Legacy polling fallback; reconciliation is the primary live synchronizer."""
        trade = await self.repo.get_active_trade(symbol, _get_execution_mode(self.settings))
        if trade is None or trade.source != "binance":
            return None
        if await self.exchange.get_position(symbol) is not None:
            return None

        if hasattr(self.exchange, "get_open_orders"):
            try:
                for order in await self.exchange.get_open_orders(symbol):
                    client_id = order.get("client_order_id") or ""
                    if not client_id.startswith("mm-"):
                        continue
                    await self.exchange.cancel_order(
                        order["id"], symbol, is_algo=order.get("is_algo", False)
                    )
            except Exception as exc:
                logger.error(f"[TradingEngine] Не удалось отменить bot-owned ордера: {exc}")

        exit_price = None
        if hasattr(self.exchange, "get_last_trade_price"):
            exit_price = await self.exchange.get_last_trade_price(symbol)
        exit_price = exit_price or trade.sl_price or trade.tp_price or trade.entry_price
        pnl = (
            (trade.entry_price - exit_price) * trade.amount
            if trade.is_short
            else (exit_price - trade.entry_price) * trade.amount
        )
        await self.repo.close_trade(trade, exit_price, pnl, update_portfolio=False)
        event = _format_position_closed(
            symbol=symbol,
            side="buy" if trade.is_short else "sell",
            price=exit_price,
            amount=trade.amount,
            pnl=pnl,
            order_id=trade.exit_order_id or "reconciled",
        )
        await self._publish_notification(event, str(uuid.uuid4()))
        return event

    async def _publish_notification(
        self, event: TradingNotification, correlation_id: str
    ) -> None:
        """Persist a transport-neutral business event before Telegram consumes it."""
        EventStore(self.session).append(
            "PositionOpened" if event.kind == "position_opened" else "PositionClosed",
            {
                "kind": event.kind,
                "symbol": event.symbol,
                "side": event.side,
                "amount": str(event.amount),
                "price": str(event.price),
                "order_id": event.order_id,
                "model_id": event.model_id,
                "confidence": event.confidence,
                "sl_ok": event.sl_ok,
                "tp_ok": event.tp_ok,
                "realized_pnl": str(event.realized_pnl) if event.realized_pnl is not None else None,
                "commissions": str(event.commissions) if event.commissions is not None else None,
                "exit_reason": event.exit_reason,
            },
            correlation_id=correlation_id,
        )
        await self.session.commit()


def _get_execution_mode(settings) -> str:
    mode = getattr(settings, "TRADING_MODE", None)
    if isinstance(mode, str) and mode in {"paper", "shadow", "testnet", "mainnet"}:
        return mode
    return "shadow" if bool(getattr(settings, "SHADOW_TRADING", False)) else "testnet"


def _make_client_order_id(mode: str, stable_key: str) -> str:
    digest = hashlib.sha256(f"{mode}:{stable_key}".encode("utf-8")).hexdigest()[:28]
    return f"mm-{digest}"


def _trade_as_position(trade) -> dict:
    return {
        "symbol": trade.symbol,
        "side": "SHORT" if trade.is_short else "LONG",
        "entry_price": float(trade.entry_price),
        "amount": float(trade.amount),
    }


def _is_close_order(position: dict | None, side: str) -> bool:
    return bool(
        position
        and (
            (side == "sell" and position["side"] == "LONG")
            or (side == "buy" and position["side"] == "SHORT")
        )
    )


def _protection_prices(
    side: str, price: float, sl_pct: float, tp_pct: float
) -> tuple[float, float, str]:
    if side == "buy":
        return price * (1 - sl_pct), price * (1 + tp_pct), "sell"
    return price * (1 + sl_pct), price * (1 - tp_pct), "buy"


def _count_consecutive_losses(trades) -> int:
    losses = 0
    for trade in reversed(trades):
        if trade.pnl is None:
            continue
        if trade.pnl < 0:
            losses += 1
        else:
            break
    return losses


def _order_from_intent(intent) -> dict:
    price = float(intent.average_fill_price or intent.requested_price or 0)
    return {
        "symbol": intent.symbol,
        "side": intent.side,
        "order_id": intent.exchange_order_id,
        "client_order_id": intent.client_order_id,
        "price": price,
        "average_price": price,
        "amount": float(intent.requested_amount),
        "filled_amount": float(intent.filled_amount or intent.requested_amount),
        "commission": float(intent.commission or 0),
        "commission_asset": intent.commission_asset,
        "status": intent.status.lower(),
        "raw_status": intent.raw_status,
        "fills": [],
        "raw": intent.raw_response,
        "pnl": None,
    }


def _format_shadow_decision(symbol, side, amount, price, decision, reason) -> str:
    return (
        f"👤 <b>SHADOW · {symbol}</b>\n\n"
        f"Решение: <code>{side.upper()}</code>\n"
        f"Объём: <code>{amount:.6f}</code>\n"
        f"Ориентир: <code>{price:.2f}$</code>\n"
        f"Risk: <code>{decision.value}</code> · {reason}\n\n"
        "Ордер не отправлялся на биржу."
    )


def _format_position_opened(
    *,
    symbol: str,
    side: str,
    price: float,
    amount: float,
    protection: str,
    order_id: str,
    model_id: str | None,
    warning: str | None = None,
) -> str:
    direction = "LONG" if side == "buy" else "SHORT"
    icon = "🟢" if direction == "LONG" else "🔴"
    model_line = f"\nМодель: <code>{model_id}</code>" if model_id else ""
    warning_line = f"\n\n{warning}" if warning else ""
    return (
        f"{icon} <b>{symbol} · {direction} открыт</b>\n\n"
        f"Исполнено: <code>{amount:.6f}</code> @ <code>{price:.2f}$</code>\n"
        f"Объём: <code>{amount * price:.2f}$</code>\n"
        f"Защита: {protection}\n"
        f"Ордер: <code>{order_id}</code>{model_line}{warning_line}"
    )


def _format_position_closed(
    *,
    symbol: str,
    side: str,
    price: float,
    amount: float,
    pnl: float | None,
    order_id: str,
) -> str:
    pnl_line = f"\nPnL: <code>{pnl:+.2f}$</code>" if pnl is not None else ""
    return (
        f"⚪️ <b>{symbol} · позиция закрыта</b>\n\n"
        f"Исполнено: <code>{amount:.6f}</code> @ <code>{price:.2f}$</code>\n"
        f"Сторона закрытия: <code>{side.upper()}</code>"
        f"{pnl_line}\nОрдер: <code>{order_id}</code>"
    )


# The legacy string helpers remain above for compatibility with old imports.
# These definitions make all new engine notifications structured contracts.
def _format_position_opened(
    *, symbol: str, side: str, price: float, amount: float, protection: str,
    order_id: str, model_id: str | None, warning: str | None = None,
) -> TradingNotification:
    del warning
    parts = protection.split("·")
    return TradingNotification(
        kind="position_opened", symbol=symbol, side=side, amount=amount,
        price=price, order_id=str(order_id), model_id=model_id,
        sl_ok="✅" in parts[0], tp_ok="✅" in parts[-1],
    )


def _format_position_closed(
    *, symbol: str, side: str, price: float, amount: float,
    pnl: float | None, order_id: str, commissions: float | None = None,
) -> TradingNotification:
    return TradingNotification(
        kind="position_closed", symbol=symbol, side=side, amount=amount,
        price=price, order_id=str(order_id), realized_pnl=pnl,
        commissions=commissions,
        exit_reason="signal/reconciliation",
    )
