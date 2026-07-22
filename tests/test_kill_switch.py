import pytest
from unittest.mock import AsyncMock

from src.crud.paper import TradeRepository
from src.risk.kill_switch import KillSwitchManager, KillSwitchState, reconcile_positions


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
async def test_kill_switch_manager_states():
    manager = KillSwitchManager(FakeRedis())
    assert (await manager.get_state())[0] == KillSwitchState.NORMAL
    await manager.set_state(KillSwitchState.SAFE_MODE, "TEST", "details")
    assert await manager.is_trading_blocked()


@pytest.mark.asyncio
async def test_reconciliation_keeps_environments_separate(temp_db_session):
    repo = TradeRepository(temp_db_session)
    await repo.create_trade(
        symbol="BTC/USDT", entry_price=100, amount=1, sl_price=None,
        tp_price=None, entry_candle_time=1, environment="paper",
    )
    exchange = AsyncMock()
    exchange.get_position.return_value = {
        "symbol": "BTC/USDT", "side": "LONG", "entry_price": "100", "amount": "1"
    }
    success, details = await reconcile_positions(
        exchange, temp_db_session, ["BTC/USDT"], KillSwitchManager(FakeRedis()),
        environment="testnet",
    )
    assert not success
    assert "intent" in details
    assert await repo.get_active_trade("BTC/USDT", "paper") is not None


@pytest.mark.asyncio
async def test_reconciliation_closes_live_projection_when_exchange_flat(temp_db_session):
    repo = TradeRepository(temp_db_session)
    trade = await repo.create_trade(
        symbol="BTC/USDT", entry_price=100, amount=1, sl_price=None,
        tp_price=None, entry_candle_time=1, environment="testnet", source="binance",
        update_portfolio=False,
    )
    exchange = AsyncMock()
    exchange.get_position.return_value = None
    success, details = await reconcile_positions(
        exchange, temp_db_session, ["BTC/USDT"], KillSwitchManager(FakeRedis()),
        environment="testnet",
    )
    assert success and details is None
    assert trade.status == "CLOSED"
    assert trade.pnl is None
