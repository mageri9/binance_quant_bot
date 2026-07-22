import pytest
from unittest.mock import AsyncMock, MagicMock

from src.exchange.engine import TradingEngine
from src.risk.engine import RiskEngine
from src.risk.kill_switch import KillSwitchManager


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, *args, **kwargs):
        self.store[key] = str(value)

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.mark.asyncio
async def test_trading_engine_shadow_trading(temp_db_session):
    redis = FakeRedis()
    kill_switch = KillSwitchManager(redis)
    risk_engine = RiskEngine()

    mock_settings = MagicMock()
    mock_settings.TRADING_MODE = "shadow"
    mock_settings.PAPER_RISK_PCT = 0.10
    mock_settings.SHADOW_TRADING = True

    mock_exchange = AsyncMock()
    mock_exchange.get_balance.return_value = {"free": 1000.0, "total": 10000.0}
    mock_exchange.get_position.return_value = None

    engine = TradingEngine(
        exchange=mock_exchange,
        risk_engine=risk_engine,
        kill_switch_manager=kill_switch,
        session=temp_db_session,
        settings=mock_settings,
    )

    msg = await engine.process_signal("BTC/USDT", signal=1, latest_close=100.0)

    assert "SHADOW" in msg
    assert "не отправлялся на биржу" in msg
    mock_exchange.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_trading_engine_live_execution(temp_db_session):
    redis = FakeRedis()
    kill_switch = KillSwitchManager(redis)
    risk_engine = RiskEngine()

    mock_settings = MagicMock()
    mock_settings.TRADING_MODE = "testnet"
    mock_settings.PAPER_RISK_PCT = 0.10
    mock_settings.PAPER_SL_PCT = 0.02
    mock_settings.PAPER_TP_PCT = 0.04
    mock_settings.SHADOW_TRADING = False

    mock_exchange = AsyncMock()
    mock_exchange.get_balance.return_value = {"free": 1000.0, "total": 10000.0}
    mock_exchange.get_position.return_value = None
    mock_exchange.create_order.return_value = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 100.05,
        "amount": 10.0,
        "commission": 1.0,
        "status": "open",
        "pnl": None,
    }

    engine = TradingEngine(
        exchange=mock_exchange,
        risk_engine=risk_engine,
        kill_switch_manager=kill_switch,
        session=temp_db_session,
        settings=mock_settings,
    )

    msg = await engine.process_signal("BTC/USDT", signal=1, latest_close=100.0)

    assert "ORDER OPEN" in msg
    mock_exchange.create_order.assert_called_once()


@pytest.mark.asyncio
async def test_lost_market_order_response_never_resubmits(temp_db_session):
    redis = FakeRedis()
    kill_switch = KillSwitchManager(redis)
    settings = MagicMock()
    settings.TRADING_MODE = "testnet"
    settings.PAPER_RISK_PCT = 0.10
    settings.PAPER_SL_PCT = 0.02
    settings.PAPER_TP_PCT = 0.04
    exchange = AsyncMock()
    exchange.get_balance.return_value = {"free": 1000.0, "total": 10000.0}
    exchange.get_position.return_value = None
    exchange.create_order.side_effect = TimeoutError("response lost")
    engine = TradingEngine(exchange, RiskEngine(), kill_switch, temp_db_session, settings)

    message = await engine.process_signal("BTC/USDT", 1, 100.0, idempotency_key="lost-response")
    assert "ORDER RECOVERY REQUIRED" in message
    assert exchange.create_order.await_count == 1

    await kill_switch.set_state("NORMAL")
    exchange.get_order_by_client_id.return_value = {
        "symbol": "BTC/USDT", "side": "buy", "order_id": "42", "client_order_id": "mm-any",
        "price": 100.0, "average_price": 100.0, "amount": 10.0, "filled_amount": 0.0,
        "commission": 0.0, "status": "open", "raw": {}, "fills": [],
    }
    message = await engine.process_signal("BTC/USDT", 1, 100.0, idempotency_key="lost-response")
    assert "ORDER OPEN" in message
    assert exchange.create_order.await_count == 1
    exchange.get_order_by_client_id.assert_awaited_once()
