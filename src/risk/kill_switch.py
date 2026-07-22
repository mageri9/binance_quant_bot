import json
from decimal import Decimal
from loguru import logger
from redis.asyncio import Redis


class KillSwitchState:
    NORMAL = "NORMAL"
    SAFE_MODE = "SAFE_MODE"
    KILLED = "KILLED"


class KillSwitchManager:
    """
    Управляет аварийным состоянием бота (Kill Switch) в Redis.
    """

    def __init__(self, redis: Redis):
        self.redis = redis

    async def get_state(self) -> tuple[str, str | None, str | None]:
        state = await self.redis.get("nexus:kill_switch:state") or KillSwitchState.NORMAL
        reason = await self.redis.get("nexus:kill_switch:reason")
        details = await self.redis.get("nexus:kill_switch:details")
        return state, reason, details

    async def set_state(self, state: str, reason: str | None = None, details: str | None = None) -> None:
        if state not in [KillSwitchState.NORMAL, KillSwitchState.SAFE_MODE, KillSwitchState.KILLED]:
            raise ValueError(f"Недопустимое состояние Kill Switch: {state}")

        await self.redis.set("nexus:kill_switch:state", state)
        if reason:
            await self.redis.set("nexus:kill_switch:reason", reason)
        else:
            await self.redis.delete("nexus:kill_switch:reason")

        if details:
            await self.redis.set("nexus:kill_switch:details", details)
        else:
            await self.redis.delete("nexus:kill_switch:details")

        logger.warning(f"[KILL SWITCH] Новое состояние: {state} | Причина: {reason or 'нет'} | Детали: {details or 'нет'}")

    async def is_trading_blocked(self) -> bool:
        state, _, _ = await self.get_state()
        return state in [KillSwitchState.SAFE_MODE, KillSwitchState.KILLED]


async def reconcile_positions(
    exchange,
    db_session,
    symbols: list[str],
    kill_switch_manager: KillSwitchManager,
    *,
    environment: str | None = None,
    verify_protection: bool = False,
) -> tuple[bool, str | None]:
    """
    Восстанавливает локальную проекцию из Binance и блокирует торговлю только
    для необъяснимой или незащищённой позиции.
    """
    from datetime import datetime, timezone

    from src.crud.execution import ExecutionRepository
    from src.crud.paper import TradeRepository

    repo = TradeRepository(db_session)
    execution_repo = ExecutionRepository(db_session)
    environment = environment or (
        "testnet" if bool(getattr(exchange, "testnet", True)) else "mainnet"
    )

    unsafe = []
    actions = []

    for symbol in symbols:
        try:
            ex_pos = await exchange.get_position(symbol)
        except Exception as exc:
            unsafe.append(f"{symbol}: Binance position snapshot недоступен: {exc}")
            continue

        db_pos = await repo.get_active_trade(symbol, environment)
        recent_fills = getattr(exchange, "get_recent_fills", None)
        if callable(recent_fills):
            try:
                fills = await recent_fills(symbol)
                await execution_repo.record_fills(
                    environment, {"symbol": symbol, "fills": fills}
                )
            except Exception as exc:
                unsafe.append(f"{symbol}: Binance fills snapshot unavailable: {exc}")
                continue
        await execution_repo.upsert_position(environment, symbol, ex_pos)

        exchange_amount = _position_amount(ex_pos)
        if ex_pos is None or exchange_amount == 0:
            if db_pos is not None:
                exit_price = await _safe_last_trade_price(exchange, symbol)
                exit_price = exit_price or db_pos.entry_price
                db_pos.source = "binance"
                await repo.close_trade(
                    db_pos,
                    exit_price,
                    None,
                    update_portfolio=False,
                )
                actions.append(
                    {
                        "symbol": symbol,
                        "action": "CLOSE_STALE_DB_POSITION",
                        "trade_id": db_pos.id,
                        "exit_price": float(exit_price),
                    }
                )
            continue

        if db_pos is None:
            intent = await execution_repo.get_latest_recoverable_intent(environment, symbol)
            owned = bool(
                intent
                and intent.side == ("buy" if ex_pos["side"] == "LONG" else "sell")
            )
            db_pos = await repo.create_trade(
                symbol=symbol,
                entry_price=ex_pos.get("entry_price") or 0,
                amount=exchange_amount,
                sl_price=float(intent.sl_price) if intent and intent.sl_price is not None else None,
                tp_price=float(intent.tp_price) if intent and intent.tp_price is not None else None,
                entry_candle_time=int(
                    ex_pos.get("timestamp")
                    or datetime.now(timezone.utc).timestamp() * 1000
                ),
                is_short=(ex_pos["side"] == "SHORT"),
                source="binance",
                environment=environment,
                client_order_id=intent.client_order_id if intent else None,
                entry_order_id=intent.exchange_order_id if intent else None,
                model_id=intent.model_id if intent else None,
                update_portfolio=False,
            )
            if intent:
                await execution_repo.link_trade(intent, db_pos.id)
            actions.append(
                {
                    "symbol": symbol,
                    "action": "REBUILD_DB_POSITION",
                    "trade_id": db_pos.id,
                    "owned": owned,
                }
            )
            if not owned:
                unsafe.append(
                    f"{symbol}: позиция восстановлена, но bot-owned order intent не найден."
                )

        ex_side = ex_pos["side"]
        db_side = "SHORT" if db_pos.is_short else "LONG"

        if ex_side != db_side:
            db_pos.is_short = ex_side == "SHORT"
            unsafe.append(
                f"{symbol}: направление локальной проекции исправлено {db_side} → {ex_side}."
            )
            actions.append(
                {"symbol": symbol, "action": "SYNC_SIDE", "from": db_side, "to": ex_side}
            )

        db_amount = Decimal(str(db_pos.amount))
        if abs(exchange_amount - db_amount) > Decimal("0.00000001"):
            previous = db_pos.amount
            db_pos.amount = exchange_amount
            actions.append(
                {
                    "symbol": symbol,
                    "action": "SYNC_AMOUNT",
                    "from": float(previous),
                    "to": str(exchange_amount),
                }
            )

        if ex_pos.get("entry_price"):
            db_pos.entry_price = ex_pos["entry_price"]
        db_pos.source = "binance"
        db_pos.last_reconciled_at = datetime.now(timezone.utc)
        await db_session.commit()

        if verify_protection:
            sl_ok, tp_ok = await _get_protection_state(exchange, execution_repo, db_pos)
            if not sl_ok or not tp_ok:
                unsafe.append(
                    f"{symbol}: bot-owned защита неполная (SL={sl_ok}, TP={tp_ok})."
                )

    if unsafe:
        error_details = "\n".join(unsafe)
        await execution_repo.log_reconciliation(
            environment=environment,
            status="UNSAFE",
            actions=actions,
            details=error_details,
        )
        await kill_switch_manager.set_state(
            state=KillSwitchState.SAFE_MODE,
            reason="RECONCILIATION_UNSAFE",
            details=error_details,
        )
        return False, error_details

    await execution_repo.log_reconciliation(
        environment=environment,
        status="SYNCED",
        actions=actions,
    )
    state, reason, _ = await kill_switch_manager.get_state()
    if state == KillSwitchState.SAFE_MODE and reason in {
        "POSITION_MISMATCH",
        "RECONCILIATION_UNSAFE",
        "EXECUTION_LEDGER_WRITE_FAILED",
        "POSITION_PROJECTION_WRITE_FAILED",
    }:
        await kill_switch_manager.set_state(KillSwitchState.NORMAL)
    return True, None


async def _safe_last_trade_price(exchange, symbol: str) -> float | None:
    method = getattr(exchange, "get_last_trade_price", None)
    if not callable(method):
        return None
    try:
        value = await method(symbol)
        return float(value) if isinstance(value, (int, float)) else None
    except Exception:
        return None


def _position_amount(position: dict | None) -> Decimal:
    if not position:
        return Decimal("0")
    value = position.get("amount")
    if value is None:
        raise ValueError("position snapshot has no amount")
    return abs(Decimal(str(value)))


async def _get_protection_state(exchange, execution_repo, trade) -> tuple[bool, bool]:
    symbol = trade.symbol
    method = getattr(exchange, "get_open_orders", None)
    if not callable(method):
        return False, False
    try:
        orders = await method(trade.symbol)
    except Exception as exc:
        logger.error(f"[Reconcile] Не удалось проверить защиту {symbol}: {exc}")
        return False, False
    if not isinstance(orders, list):
        return False, False

    if not trade.client_order_id:
        return False, False
    entry_intent = await execution_repo.get_intent(trade.client_order_id)
    if entry_intent is None:
        return False, False
    expected = {
        intent.client_order_id: intent.purpose
        for intent in await execution_repo.get_protection_intents(entry_intent.id)
    }
    sl_ok = False
    tp_ok = False
    for order in orders:
        client_id = str(order.get("client_order_id") or "")
        purpose = expected.get(client_id)
        if purpose is None:
            continue
        order_type = str(order.get("type") or "").upper()
        sl_ok = sl_ok or (purpose == "STOP_LOSS" and order_type in {"STOP", "STOP_MARKET"})
        tp_ok = tp_ok or (purpose == "TAKE_PROFIT" and order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"})
    return sl_ok, tp_ok
