import pytest
from unittest.mock import AsyncMock

from src.risk.kill_switch import KillSwitchManager, KillSwitchState, reconcile_positions
from src.crud.paper import TradeRepository


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
    redis = FakeRedis()
    manager = KillSwitchManager(redis)

    state, reason, details = await manager.get_state()
    assert state == KillSwitchState.NORMAL
    assert not await manager.is_trading_blocked()

    await manager.set_state(
        KillSwitchState.SAFE_MODE, "TEST_ERR", "Details of test error"
    )
    state, reason, details = await manager.get_state()
    assert state == KillSwitchState.SAFE_MODE
    assert reason == "TEST_ERR"
    assert details == "Details of test error"
    assert await manager.is_trading_blocked()


@pytest.mark.asyncio
async def test_reconciliation_perfect_sync(temp_db_session):
    redis = FakeRedis()
    manager = KillSwitchManager(redis)

    repo = TradeRepository(temp_db_session)
    await repo.create_trade(
        symbol="BTC/USDT",
        entry_price=100.0,
        amount=1.5,
        sl_price=None,
        tp_price=None,
        entry_candle_time=1000,
        is_short=False,
    )

    mock_exchange = AsyncMock()
    mock_exchange.get_position.return_value = {
        "symbol": "BTC/USDT",
        "side": "LONG",
        "entry_price": 100.0,
        "amount": 1.5,
    }

    success, err = await reconcile_positions(
        exchange=mock_exchange,
        db_session=temp_db_session,
        symbols=["BTC/USDT"],
        kill_switch_manager=manager,
    )

    assert success is True
    assert err is None
    assert not await manager.is_trading_blocked()


@pytest.mark.asyncio
async def test_reconciliation_detects_missing_exchange_position(temp_db_session):
    redis = FakeRedis()
    manager = KillSwitchManager(redis)

    repo = TradeRepository(temp_db_session)
    await repo.create_trade(
        symbol="BTC/USDT",
        entry_price=100.0,
        amount=1.5,
        sl_price=None,
        tp_price=None,
        entry_candle_time=1000,
        is_short=False,
    )

    mock_exchange = AsyncMock()
    mock_exchange.get_position.return_value = None

    success, err = await reconcile_positions(
        exchange=mock_exchange,
        db_session=temp_db_session,
        symbols=["BTC/USDT"],
        kill_switch_manager=manager,
    )

    assert success is False
    assert "позиция отсутствует" in err

    state, reason, details = await manager.get_state()
    assert state == KillSwitchState.SAFE_MODE
    assert reason == "POSITION_MISMATCH"