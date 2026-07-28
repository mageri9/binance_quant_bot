import asyncio
import numpy as np
import pandas as pd
import pytest
from src.db import AsyncSessionFactory, Kline, init_db
from src.event_bus import AsyncEventBus, CandleClosedEvent, SignalEmittedEvent, OrderApprovedEvent
from src.exchange.paper import PaperExchange
from src.services.risk_service import RiskService
from src.services.strategy_service import StrategyService
from src.strategy.features import add_features


@pytest.mark.asyncio
async def test_full_pipeline_event_flow():
    await init_db()
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


@pytest.mark.asyncio
async def test_risk_service_uses_signal_atr_for_protection_prices():
    await init_db()
    bus = AsyncEventBus()
    exchange = PaperExchange(initial_balance=10000.0)
    approved_orders = []

    async def on_order_approved(event: OrderApprovedEvent):
        approved_orders.append(event)

    bus.subscribe(OrderApprovedEvent, on_order_approved)
    RiskService(bus, exchange)

    await bus.publish(SignalEmittedEvent(
        symbol="ETH/USDT", timeframe="1h", signal=1, expected_return=0.005,
        close_price=100.0, open_time=1000, atr=3.0,
    ))

    assert len(approved_orders) == 1
    assert approved_orders[0].sl_price == 97.0
    assert approved_orders[0].tp_price == 104.5


@pytest.mark.asyncio
async def test_strategy_service_publishes_atr_from_latest_features(tmp_path):
    await init_db()
    prices = np.concatenate([np.linspace(100, 70, 26), np.linspace(70, 74, 4)])
    symbol = "BTC/USDT"
    timeframe = "1h"
    candles = pd.DataFrame({
        "open_time": np.arange(30) * 3_600_000,
        "open": prices - 0.5,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": np.full(30, 1000.0),
    })
    expected_atr = float(add_features(candles).iloc[-1]["atr"])

    async with AsyncSessionFactory() as session:
        session.add_all([
            Kline(symbol=symbol, timeframe=timeframe, **candle)
            for candle in candles.to_dict("records")
        ])
        await session.commit()

    bus = AsyncEventBus()
    signals = []

    async def on_signal(event: SignalEmittedEvent):
        signals.append(event)

    bus.subscribe(SignalEmittedEvent, on_signal)
    StrategyService(bus, model_dir=str(tmp_path))

    await bus.publish(CandleClosedEvent(
        symbol=symbol, timeframe=timeframe, open_time=int(candles.iloc[-1]["open_time"]),
        open=float(candles.iloc[-1]["open"]), high=float(candles.iloc[-1]["high"]),
        low=float(candles.iloc[-1]["low"]), close=float(candles.iloc[-1]["close"]),
        volume=float(candles.iloc[-1]["volume"]),
    ))

    assert len(signals) == 1
    assert signals[0].atr == expected_atr


@pytest.mark.asyncio
async def test_risk_service_rejects_concurrent_signal_for_pending_symbol():
    await init_db()
    bus = AsyncEventBus()
    exchange = PaperExchange(initial_balance=10000.0)
    approved_orders = []

    async def on_order_approved(event: OrderApprovedEvent):
        approved_orders.append(event)
        await asyncio.sleep(0)

    bus.subscribe(OrderApprovedEvent, on_order_approved)
    RiskService(bus, exchange)
    event = SignalEmittedEvent(
        symbol="BTC/USDT", timeframe="1h", signal=1, expected_return=0.005,
        close_price=50000.0, open_time=1000,
    )

    await asyncio.gather(bus.publish(event), bus.publish(event))

    assert len(approved_orders) == 1
