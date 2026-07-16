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

    assert "SHADOW TRADING" in msg
    assert "НЕ отправлен на биржу" in msg
    mock_exchange.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_trading_engine_live_execution(temp_db_session):
    redis = FakeRedis()
    kill_switch = KillSwitchManager(redis)
    risk_engine = RiskEngine()

    mock_settings = MagicMock()
    mock_settings.PAPER_RISK_PCT = 0.10
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

    assert "Исполнен ордер BUY" in msg
    mock_exchange.create_order.assert_called_once()