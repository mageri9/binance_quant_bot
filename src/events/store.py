"""Database-backed event contracts and transactional outbox helpers."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import OutboxEvent, ProcessedEvent

EVENT_TYPES = frozenset({
    "CandleClosed", "PredictionGenerated", "RiskApproved", "RiskRejected",
    "OrderRequested", "OrderAccepted", "OrderPartiallyFilled", "OrderFilled",
    "OrderRejected", "PositionOpened", "PositionUpdated", "PositionClosed",
    "ProtectionPlaced", "ProtectionFailed", "ReconciliationCompleted",
    "ModelTrained", "ModelPromoted", "ModelRejected",
})


class EventStore:
    """Writes events in the caller's transaction; delivery can remain in-process."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        correlation_id: str,
        causation_id: str | None = None,
        binance_event_id: str | None = None,
        payload_version: int = 1,
        event_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> OutboxEvent:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {event_type}")
        event = OutboxEvent(
            event_id=event_id or str(uuid4()),
            event_type=event_type,
            correlation_id=correlation_id,
            causation_id=causation_id,
            occurred_at=occurred_at or datetime.now(timezone.utc),
            payload_version=payload_version,
            binance_event_id=binance_event_id,
            payload=payload,
        )
        self.session.add(event)
        return event

    async def mark_processed(self, consumer: str, event_id: str) -> bool:
        """Claim an event for one consumer. Returns False for a replay."""
        try:
            # A duplicate must not roll back unrelated projection updates in the
            # surrounding transaction.
            async with self.session.begin_nested():
                self.session.add(ProcessedEvent(consumer=consumer, event_id=event_id))
                await self.session.flush()
        except IntegrityError:
            return False
        return True

    async def unpublished(self, limit: int = 100) -> list[OutboxEvent]:
        result = await self.session.execute(
            select(OutboxEvent)
            .where(OutboxEvent.published_at.is_(None))
            .order_by(OutboxEvent.id)
            .limit(limit)
        )
        return list(result.scalars())
