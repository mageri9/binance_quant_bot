import pytest
from src.event_bus import AsyncEventBus, SignalEmittedEvent, OrderApprovedEvent
from src.exchange.paper import PaperExchange
from src.services.risk_service import RiskService


@pytest.mark.asyncio
async def test_full_pipeline_event_flow():
    bus = AsyncEventBus()
    exchange = PaperExchange(initial_balance=10000.0)

    approved_orders = []

    async def on_order_approved(event: OrderApprovedEvent):
        approved_orders.append(event)

    bus.subscribe(OrderApprovedEvent, on_order_approved)
    risk_service = RiskService(bus, exchange)

    await bus.publish(SignalEmittedEvent(
        symbol="BTC/USDT", timeframe="1h", signal=1, expected_return=0.005, close_price=50000.0, open_time=1000
    ))

    assert len(approved_orders) == 1
    assert approved_orders[0].symbol == "BTC/USDT"
    assert approved_orders[0].side == "buy"
    assert approved_orders[0].amount > 0