import pytest
from sqlalchemy import select

from src.crud.execution import ExecutionRepository
from src.db.models import ExchangeOrder, OutboxEvent, ProcessedEvent
from src.events import EventStore


@pytest.mark.asyncio
async def test_order_request_is_written_to_outbox_in_the_intent_transaction(temp_db_session):
    repo = ExecutionRepository(temp_db_session)
    intent, created = await repo.create_intent(
        correlation_id="00000000-0000-0000-0000-000000000010",
        client_order_id="mm-outbox-order",
        environment="testnet",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        requested_amount="1",
        requested_price=None,
        sl_price=None,
        tp_price=None,
    )

    events = (await temp_db_session.execute(select(OutboxEvent))).scalars().all()
    assert created
    assert [(event.event_type, event.correlation_id) for event in events] == [
        ("OrderRequested", intent.correlation_id)
    ]
    assert events[0].payload_version == 1
    assert events[0].event_id


@pytest.mark.asyncio
async def test_exchange_order_projection_and_event_are_idempotent(temp_db_session):
    repo = ExecutionRepository(temp_db_session)
    await repo.create_intent(
        correlation_id="00000000-0000-0000-0000-000000000011",
        client_order_id="mm-outbox-fill",
        environment="testnet",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        requested_amount="1",
        requested_price=None,
        sl_price=None,
        tp_price=None,
    )
    payload = {
        "e": "ORDER_TRADE_UPDATE", "E": 200,
        "o": {"c": "mm-outbox-fill", "i": 9, "T": 200, "X": "FILLED", "x": "TRADE",
              "z": "1", "ap": "101", "l": "1", "L": "101", "t": 12, "S": "BUY"},
    }
    await repo.apply_user_stream_event("testnet", payload)
    await repo.apply_user_stream_event("testnet", payload)

    orders = (await temp_db_session.execute(select(ExchangeOrder))).scalars().all()
    events = (await temp_db_session.execute(select(OutboxEvent))).scalars().all()
    assert len(orders) == 1
    assert orders[0].binance_order_id == "9"
    assert [event.event_type for event in events] == ["OrderRequested", "OrderFilled"]


@pytest.mark.asyncio
async def test_processed_event_is_idempotent_without_rolling_back_transaction(temp_db_session):
    store = EventStore(temp_db_session)
    assert await store.mark_processed("positions", "00000000-0000-0000-0000-000000000012")
    await temp_db_session.commit()
    assert not await store.mark_processed("positions", "00000000-0000-0000-0000-000000000012")
    await temp_db_session.commit()

    processed = (await temp_db_session.execute(select(ProcessedEvent))).scalars().all()
    assert len(processed) == 1
