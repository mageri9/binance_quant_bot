from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy import delete, select

from src.db import AsyncSessionFactory, Trade, init_db
from src.event_bus import (
    AsyncEventBus,
    CandleClosedEvent,
    ErrorEvent,
    OrderApprovedEvent,
    OrderExecutedEvent,
    SignalEmittedEvent,
)
from src.exchange.paper import PaperExchange
from src.services.execution_service import ExecutionService
from src.services.market import MarketService
from src.services.risk_service import RiskService


@pytest.mark.asyncio
async def test_market_service_publishes_only_new_closed_candle():
    await init_db()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    exchange = type("Exchange", (), {
        "get_klines": AsyncMock(return_value=pd.DataFrame([{
            "open_time": now_ms - 120_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }]))
    })()
    bus = AsyncEventBus()
    events = []

    async def on_candle_closed(event: CandleClosedEvent):
        events.append(event)

    bus.subscribe(CandleClosedEvent, on_candle_closed)
    service = MarketService(bus, exchange)

    await service.fetch_and_publish_klines("BTC/USDT", "1m")
    await service.fetch_and_publish_klines("BTC/USDT", "1m")

    assert len(events) == 1


@pytest.mark.asyncio
async def test_risk_service_rejects_entry_for_open_position():
    await init_db()
    async with AsyncSessionFactory() as session:
        await session.execute(delete(Trade))
        session.add(Trade(
            symbol="BTC/USDT",
            status="OPEN",
            side="LONG",
            entry_price=50_000.0,
            amount=0.01,
            mode="paper",
        ))
        await session.commit()

    service = RiskService(AsyncEventBus(), PaperExchange())
    order = await service._approve_signal(SignalEmittedEvent(
        symbol="BTC/USDT",
        timeframe="1h",
        signal=1,
        expected_return=0.005,
        close_price=50_000.0,
        open_time=1_000,
    ))

    assert order is None

    async with AsyncSessionFactory() as session:
        await session.execute(delete(Trade))
        await session.commit()


@pytest.mark.asyncio
async def test_execution_records_trade_and_emits_error_when_stop_orders_fail():
    class StopOrderFailureExchange(PaperExchange):
        async def create_stop_orders(self, **_kwargs):
            raise RuntimeError("stop order creation failed")

    await init_db()
    async with AsyncSessionFactory() as session:
        await session.execute(delete(Trade))
        await session.commit()
    bus = AsyncEventBus()
    errors = []
    executed_orders = []

    async def on_error(event: ErrorEvent):
        errors.append(event)

    async def on_order_executed(event: OrderExecutedEvent):
        executed_orders.append(event)

    bus.subscribe(ErrorEvent, on_error)
    bus.subscribe(OrderExecutedEvent, on_order_executed)
    service = ExecutionService(bus, StopOrderFailureExchange())

    await service.on_order_approved(OrderApprovedEvent(
        symbol="BTC/USDT",
        side="buy",
        amount=0.01,
        price=50_000.0,
        sl_price=49_000.0,
        tp_price=52_000.0,
        reason="test",
    ))

    async with AsyncSessionFactory() as session:
        trades = (await session.execute(
            select(Trade).where(Trade.symbol == "BTC/USDT")
        )).scalars().all()

    assert len(trades) == 1
    assert len(errors) == 1
    assert not executed_orders

    async with AsyncSessionFactory() as session:
        await session.execute(delete(Trade))
        await session.commit()
