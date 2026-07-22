import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    ExchangeFill,
    ExchangeEvent,
    ExchangeOrder,
    BalanceSnapshot,
    OrderIntent,
    PositionSnapshot,
    ReconciliationRun,
)
from src.events import EventStore


def as_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def json_safe(value: Any) -> Any:
    """Convert exchange payloads to values accepted by SQL JSON columns."""
    return json.loads(json.dumps(value, default=str))


class ExecutionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_intent(self, client_order_id: str) -> OrderIntent | None:
        result = await self.session.execute(
            select(OrderIntent).where(OrderIntent.client_order_id == client_order_id)
        )
        return result.scalar_one_or_none()

    async def create_intent(
        self,
        *,
        correlation_id: str,
        client_order_id: str,
        environment: str,
        symbol: str,
        side: str,
        order_type: str,
        requested_amount: float,
        requested_price: float | None,
        sl_price: float | None,
        tp_price: float | None,
        model_id: str | None = None,
        prediction_id: int | None = None,
        purpose: str = "ENTRY",
        parent_intent_id: int | None = None,
        reduce_only: bool = False,
    ) -> tuple[OrderIntent, bool]:
        existing = await self.get_intent(client_order_id)
        if existing is not None:
            if existing.symbol != symbol or existing.side != side:
                raise ValueError(
                    f"client_order_id {client_order_id} already belongs to another order"
                )
            return existing, False

        intent = OrderIntent(
            correlation_id=correlation_id,
            client_order_id=client_order_id,
            environment=environment,
            symbol=symbol,
            side=side,
            order_type=order_type,
            status="PENDING",
            purpose=purpose,
            parent_intent_id=parent_intent_id,
            reduce_only=reduce_only,
            requested_amount=as_decimal(requested_amount),
            requested_price=as_decimal(requested_price),
            sl_price=as_decimal(sl_price),
            tp_price=as_decimal(tp_price),
            model_id=model_id,
            prediction_id=prediction_id,
        )
        self.session.add(intent)
        EventStore(self.session).append(
            "OrderRequested",
            _intent_payload(intent),
            correlation_id=correlation_id,
        )
        await self.session.commit()
        await self.session.refresh(intent)
        return intent, True

    async def apply_exchange_order(
        self, intent: OrderIntent, order: dict
    ) -> OrderIntent:
        now = datetime.now(timezone.utc)
        status = str(order.get("status") or "submitted").upper()
        intent.status = {
            "CLOSED": "FILLED",
            "FILLED": "FILLED",
            "OPEN": "SUBMITTED",
            "NEW": "SUBMITTED",
            "CANCELED": "CANCELED",
            "CANCELLED": "CANCELED",
            "REJECTED": "REJECTED",
        }.get(status, status)
        intent.exchange_order_id = _string_or_none(order.get("order_id"))
        intent.raw_status = _string_or_none(order.get("raw_status") or order.get("status"))
        # Zero is meaningful: never replace it with the requested amount.
        filled = order.get("filled_amount")
        intent.filled_amount = as_decimal(filled) if filled is not None else None
        intent.average_fill_price = as_decimal(order.get("average_price") or order.get("price"))
        intent.commission = as_decimal(order.get("commission") or 0)
        intent.commission_asset = _string_or_none(order.get("commission_asset"))
        intent.raw_response = json_safe(order.get("raw") or order)
        exchange_time = order.get("timestamp")
        if exchange_time is not None:
            intent.exchange_update_time = int(exchange_time)
        intent.submitted_at = intent.submitted_at or now
        if intent.status == "FILLED":
            intent.filled_at = now
        intent.error = None

        await self._record_order_projection(intent, order)
        self._append_order_event(intent, order)

        await self.session.commit()
        await self.session.refresh(intent)
        return intent

    async def _record_order_projection(self, intent: OrderIntent, order: dict) -> None:
        binance_order_id = _string_or_none(order.get("order_id"))
        if not binance_order_id:
            return
        result = await self.session.execute(
            select(ExchangeOrder).where(
                ExchangeOrder.environment == intent.environment,
                ExchangeOrder.binance_order_id == binance_order_id,
            )
        )
        projection = result.scalar_one_or_none()
        if projection is None:
            projection = ExchangeOrder(
                environment=intent.environment,
                binance_order_id=binance_order_id,
                symbol=intent.symbol,
                status=intent.status,
            )
            self.session.add(projection)
        projection.order_intent_id = intent.id
        projection.client_order_id = intent.client_order_id
        projection.status = intent.status
        projection.exchange_update_time = intent.exchange_update_time
        projection.raw_payload = json_safe(order.get("raw") or order)

    def _append_order_event(self, intent: OrderIntent, order: dict) -> None:
        event_type = _order_event_type(intent.status)
        if event_type is None:
            return
        EventStore(self.session).append(
            event_type,
            _intent_payload(intent),
            correlation_id=intent.correlation_id,
            binance_event_id=_string_or_none(order.get("event_id")),
        )

    async def mark_failed(self, intent: OrderIntent, error: str) -> None:
        intent.status = "FAILED"
        intent.error = error
        EventStore(self.session).append(
            "OrderRejected", _intent_payload(intent), correlation_id=intent.correlation_id
        )
        await self.session.commit()

    async def mark_submitted(self, intent: OrderIntent) -> None:
        """Durably record the ambiguous boundary immediately before HTTP I/O."""
        intent.status = "SUBMITTED"
        intent.submitted_at = intent.submitted_at or datetime.now(timezone.utc)
        intent.error = None
        EventStore(self.session).append(
            "OrderAccepted", _intent_payload(intent), correlation_id=intent.correlation_id
        )
        await self.session.commit()

    async def mark_submission_uncertain(self, intent: OrderIntent, error: str) -> None:
        """Keep an ambiguous market request recoverable; never mark it rejected."""
        intent.status = "SUBMITTED"
        intent.submitted_at = intent.submitted_at or datetime.now(timezone.utc)
        intent.error = error
        await self.session.commit()

    async def link_trade(self, intent: OrderIntent, trade_id: int) -> None:
        intent.trade_id = trade_id
        await self.session.commit()

    async def get_latest_recoverable_intent(
        self, environment: str, symbol: str
    ) -> OrderIntent | None:
        result = await self.session.execute(
            select(OrderIntent)
            .where(
                OrderIntent.environment == environment,
                OrderIntent.symbol == symbol,
                OrderIntent.purpose == "ENTRY",
                OrderIntent.status.in_(["PENDING", "SUBMITTED", "PARTIALLY_FILLED", "FILLED"]),
            )
            .order_by(OrderIntent.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_protection_intents(self, parent_intent_id: int) -> list[OrderIntent]:
        result = await self.session.execute(
            select(OrderIntent).where(
                OrderIntent.parent_intent_id == parent_intent_id,
                OrderIntent.purpose.in_(["STOP_LOSS", "TAKE_PROFIT"]),
            )
        )
        return list(result.scalars().all())

    async def record_fills(self, environment: str, order: dict) -> list[ExchangeFill]:
        fills = order.get("fills") or []
        if not fills and order.get("filled_amount"):
            fills = [
                {
                    "trade_id": None,
                    "order_id": order.get("order_id"),
                    "client_order_id": order.get("client_order_id"),
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "price": order.get("average_price") or order.get("price"),
                    "amount": order.get("filled_amount") or order.get("amount"),
                    "commission": order.get("commission") or 0,
                    "commission_asset": order.get("commission_asset"),
                    "timestamp": order.get("timestamp"),
                    "raw": order.get("raw"),
                }
            ]

        recorded = []
        for index, fill in enumerate(fills):
            fill_key = _fill_key(environment, order, fill, index)
            existing = await self.session.execute(
                select(ExchangeFill).where(ExchangeFill.fill_key == fill_key)
            )
            if existing.scalar_one_or_none() is not None:
                continue

            row = ExchangeFill(
                fill_key=fill_key,
                environment=environment,
                symbol=fill.get("symbol") or order.get("symbol"),
                side=str(fill.get("side") or order.get("side") or "").lower(),
                exchange_trade_id=_string_or_none(fill.get("trade_id") or fill.get("id")),
                exchange_order_id=_string_or_none(fill.get("order_id") or order.get("order_id")),
                client_order_id=_string_or_none(
                    fill.get("client_order_id") or order.get("client_order_id")
                ),
                price=as_decimal(fill.get("price") or order.get("average_price") or 0),
                amount=as_decimal(fill.get("amount") or fill.get("filled") or 0),
                commission=as_decimal(fill.get("commission") or 0),
                commission_asset=_string_or_none(fill.get("commission_asset")),
                realized_pnl=as_decimal(fill.get("realized_pnl")),
                exchange_time=fill.get("timestamp"),
                raw_payload=json_safe(fill.get("raw") or fill),
            )
            self.session.add(row)
            recorded.append(row)

        if recorded:
            await self.session.commit()
            for row in recorded:
                await self.session.refresh(row)
        return recorded

    async def upsert_position(
        self, environment: str, symbol: str, position: dict | None
    ) -> PositionSnapshot:
        result = await self.session.execute(
            select(PositionSnapshot).where(
                PositionSnapshot.environment == environment,
                PositionSnapshot.symbol == symbol,
            )
        )
        snapshot = result.scalar_one_or_none()
        if snapshot is None:
            snapshot = PositionSnapshot(
                environment=environment,
                symbol=symbol,
                amount=Decimal("0"),
            )
            self.session.add(snapshot)

        payload = position or {}
        snapshot.side = payload.get("side")
        snapshot.amount = as_decimal(payload.get("amount") or 0)
        snapshot.entry_price = as_decimal(payload.get("entry_price"))
        snapshot.mark_price = as_decimal(payload.get("mark_price"))
        snapshot.unrealized_pnl = as_decimal(payload.get("unrealized_pnl"))
        snapshot.leverage = as_decimal(payload.get("leverage"))
        snapshot.exchange_update_time = payload.get("timestamp")
        snapshot.raw_payload = json_safe(payload.get("raw") or payload)
        snapshot.reconciled_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(snapshot)
        return snapshot

    async def upsert_balance(
        self, environment: str, asset: str, payload: dict, event_time: int | None
    ) -> BalanceSnapshot:
        result = await self.session.execute(
            select(BalanceSnapshot).where(
                BalanceSnapshot.environment == environment,
                BalanceSnapshot.asset == asset,
            )
        )
        snapshot = result.scalar_one_or_none()
        if snapshot is None:
            snapshot = BalanceSnapshot(environment=environment, asset=asset, wallet_balance=Decimal("0"))
            self.session.add(snapshot)
        if snapshot.update_time is not None and event_time is not None and event_time < snapshot.update_time:
            return snapshot
        snapshot.wallet_balance = as_decimal(payload.get("wb") or payload.get("walletBalance") or 0)
        snapshot.cross_wallet_balance = as_decimal(payload.get("cw") or payload.get("crossWalletBalance"))
        snapshot.available_balance = as_decimal(payload.get("ab") or payload.get("availableBalance"))
        snapshot.update_time = event_time
        snapshot.raw_payload = json_safe(payload)
        snapshot.reconciled_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(snapshot)
        return snapshot

    async def apply_user_stream_event(self, environment: str, payload: dict) -> dict:
        """Persist one Binance User Data Stream event exactly once.

        The inbox is committed before projections so a reconnect cannot replay a
        trade into the ledger. Older order snapshots are retained in the inbox
        but never overwrite newer cumulative quantities.
        """
        event_type = str(payload.get("e") or payload.get("eventType") or "UNKNOWN")
        event_time = payload.get("E") or payload.get("T") or payload.get("eventTime")
        event_key = _event_key(environment, payload)
        event = ExchangeEvent(
            event_key=event_key,
            environment=environment,
            event_type=event_type,
            event_time=int(event_time) if event_time is not None else None,
            raw_payload=json_safe(payload),
        )
        self.session.add(event)
        try:
            await self.session.commit()
        except IntegrityError:
            await self.session.rollback()
            return {"duplicate": True, "order_changed": False, "symbols": []}

        if event_type == "ORDER_TRADE_UPDATE":
            return await self._apply_order_trade_update(environment, payload, event_time)
        if event_type == "ACCOUNT_UPDATE":
            return await self._apply_account_update(environment, payload, event_time)
        return {"duplicate": False, "order_changed": False, "symbols": []}

    async def _apply_order_trade_update(
        self, environment: str, payload: dict, event_time: int | None
    ) -> dict:
        order = payload.get("o") or {}
        client_order_id = order.get("c")
        if not client_order_id:
            return {"duplicate": False, "order_changed": False, "symbols": []}
        intent = await self.get_intent(str(client_order_id))
        if intent is None:
            return {"duplicate": False, "order_changed": False, "symbols": [order.get("s")]}
        order_time = order.get("T") or event_time
        if (
            intent.exchange_update_time is not None
            and order_time is not None
            and int(order_time) < intent.exchange_update_time
        ):
            return {"duplicate": False, "order_changed": False, "symbols": [intent.symbol]}

        raw_status = str(order.get("X") or "UNKNOWN")
        normalized = {
            "NEW": "SUBMITTED",
            "PARTIALLY_FILLED": "PARTIALLY_FILLED",
            "FILLED": "FILLED",
            "CANCELED": "CANCELED",
            "REJECTED": "REJECTED",
            "EXPIRED": "EXPIRED",
        }.get(raw_status, raw_status)
        intent.status = normalized
        intent.exchange_order_id = _string_or_none(order.get("i"))
        intent.raw_status = raw_status
        intent.filled_amount = as_decimal(order.get("z"))
        intent.average_fill_price = as_decimal(order.get("ap"))
        intent.commission = as_decimal(order.get("n") or 0)
        intent.commission_asset = _string_or_none(order.get("N"))
        intent.raw_response = json_safe(payload)
        intent.submitted_at = intent.submitted_at or datetime.now(timezone.utc)
        intent.exchange_update_time = int(order_time) if order_time is not None else None
        if normalized == "FILLED":
            intent.filled_at = datetime.now(timezone.utc)
        await self._record_order_projection(intent, {
            "order_id": order.get("i"), "event_id": _event_key(environment, payload),
            "raw": payload,
        })
        self._append_order_event(intent, {"event_id": _event_key(environment, payload)})
        await self.session.commit()

        last_amount = as_decimal(order.get("l")) or Decimal("0")
        if str(order.get("x")) == "TRADE" and last_amount > 0:
            fill = {
                "trade_id": order.get("t"),
                "order_id": order.get("i"),
                "client_order_id": client_order_id,
                "symbol": intent.symbol,
                "side": order.get("S"),
                "price": order.get("L"),
                "amount": str(last_amount),
                "commission": order.get("n") or 0,
                "commission_asset": order.get("N"),
                "realized_pnl": order.get("rp"),
                "timestamp": order_time,
                "raw": payload,
            }
            await self.record_fills(environment, {"fills": [fill], "symbol": intent.symbol})
        return {"duplicate": False, "order_changed": True, "symbols": [intent.symbol]}

    async def _apply_account_update(
        self, environment: str, payload: dict, event_time: int | None
    ) -> dict:
        account = payload.get("a") or {}
        for balance in account.get("B") or []:
            asset = balance.get("a")
            if asset:
                await self.upsert_balance(environment, str(asset), balance, event_time)
        symbols: list[str] = []
        for position in account.get("P") or []:
            symbol = position.get("s")
            if not symbol:
                continue
            amount = as_decimal(position.get("pa")) or Decimal("0")
            canonical_symbol = _canonical_symbol(str(symbol))
            symbols.append(canonical_symbol)
            position_payload = None
            if amount != 0:
                position_payload = {
                    "side": "LONG" if amount > 0 else "SHORT",
                    "amount": str(abs(amount)),
                    "entry_price": position.get("ep"),
                    "unrealized_pnl": position.get("up"),
                    "timestamp": event_time,
                    "raw": position,
                }
            await self.upsert_position(environment, canonical_symbol, position_payload)
        return {"duplicate": False, "order_changed": False, "symbols": symbols}

    async def log_reconciliation(
        self,
        *,
        environment: str,
        status: str,
        actions: list[dict],
        details: str | None = None,
    ) -> ReconciliationRun:
        row = ReconciliationRun(
            environment=environment,
            status=status,
            actions=json_safe(actions),
            details=details,
        )
        self.session.add(row)
        await self.session.commit()
        await self.session.refresh(row)
        return row


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _fill_key(environment: str, order: dict, fill: dict, index: int) -> str:
    trade_id = fill.get("trade_id") or fill.get("id")
    if trade_id is not None:
        return f"{environment}:trade:{trade_id}"
    order_id = fill.get("order_id") or order.get("order_id") or "unknown"
    # REST aggregate snapshots are not fills. Include cumulative quantity and
    # timestamp so a later partial/final snapshot is not discarded.
    cumulative = fill.get("amount") or fill.get("filled") or order.get("filled_amount") or "0"
    timestamp = fill.get("timestamp") or order.get("timestamp") or "unknown"
    return f"{environment}:order:{order_id}:aggregate:{index}:{cumulative}:{timestamp}"


def _event_key(environment: str, payload: dict) -> str:
    event_type = str(payload.get("e") or payload.get("eventType") or "UNKNOWN")
    event_time = payload.get("E") or payload.get("T") or payload.get("eventTime") or "unknown"
    order = payload.get("o") or {}
    if event_type == "ORDER_TRADE_UPDATE":
        return ":".join(
            [
                environment,
                event_type,
                str(order.get("i") or "unknown"),
                str(order.get("t") or order.get("x") or "unknown"),
                str(event_time),
            ]
        )
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return f"{environment}:{event_type}:{event_time}:{digest}"


def _canonical_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


def _order_event_type(status: str) -> str | None:
    return {
        "SUBMITTED": "OrderAccepted",
        "PARTIALLY_FILLED": "OrderPartiallyFilled",
        "FILLED": "OrderFilled",
        "REJECTED": "OrderRejected",
    }.get(status)


def _intent_payload(intent: OrderIntent) -> dict[str, Any]:
    return {
        "order_intent_id": intent.id,
        "client_order_id": intent.client_order_id,
        "environment": intent.environment,
        "symbol": intent.symbol,
        "side": intent.side,
        "status": intent.status,
        "exchange_order_id": intent.exchange_order_id,
        "filled_amount": str(intent.filled_amount) if intent.filled_amount is not None else None,
    }
