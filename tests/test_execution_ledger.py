import pytest

from src.crud.execution import ExecutionRepository


@pytest.mark.asyncio
async def test_user_stream_events_are_idempotent_and_ignore_old_snapshots(temp_db_session):
    repo = ExecutionRepository(temp_db_session)
    intent, _ = await repo.create_intent(
        correlation_id="00000000-0000-0000-0000-000000000001",
        client_order_id="mm-test-order",
        environment="testnet",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        requested_amount="1",
        requested_price="100",
        sl_price=None,
        tp_price=None,
    )
    filled = {
        "e": "ORDER_TRADE_UPDATE", "E": 200,
        "o": {"c": "mm-test-order", "i": 7, "T": 200, "X": "FILLED", "x": "TRADE",
              "z": "1", "ap": "101", "l": "1", "L": "101", "t": 9, "S": "BUY", "n": "0.1", "N": "USDT"},
    }
    assert not (await repo.apply_user_stream_event("testnet", filled))["duplicate"]
    assert (await repo.apply_user_stream_event("testnet", filled))["duplicate"]

    old = {**filled, "E": 100, "o": {**filled["o"], "T": 100, "X": "NEW", "x": "NEW", "z": "0"}}
    await repo.apply_user_stream_event("testnet", old)
    assert intent.status == "FILLED"
    assert str(intent.filled_amount) == "1.000000000000000000"


@pytest.mark.asyncio
async def test_partial_and_final_fills_are_recorded_separately(temp_db_session):
    repo = ExecutionRepository(temp_db_session)
    intent, _ = await repo.create_intent(
        correlation_id="00000000-0000-0000-0000-000000000002",
        client_order_id="mm-partial-order",
        environment="testnet",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        requested_amount="2",
        requested_price="100",
        sl_price=None,
        tp_price=None,
    )
    partial = {
        "e": "ORDER_TRADE_UPDATE", "E": 100,
        "o": {"c": "mm-partial-order", "i": 8, "T": 100, "X": "PARTIALLY_FILLED", "x": "TRADE",
              "z": "1", "ap": "100", "l": "1", "L": "100", "t": 10, "S": "BUY", "n": "0.1", "N": "USDT"},
    }
    final = {
        "e": "ORDER_TRADE_UPDATE", "E": 200,
        "o": {"c": "mm-partial-order", "i": 8, "T": 200, "X": "FILLED", "x": "TRADE",
              "z": "2", "ap": "101", "l": "1", "L": "102", "t": 11, "S": "BUY", "n": "0.1", "N": "USDT"},
    }
    await repo.apply_user_stream_event("testnet", partial)
    assert intent.status == "PARTIALLY_FILLED"
    await repo.apply_user_stream_event("testnet", final)
    assert intent.status == "FILLED"

    from sqlalchemy import select
    from src.db.models import ExchangeFill
    fills = (await temp_db_session.execute(select(ExchangeFill))).scalars().all()
    assert {fill.exchange_trade_id for fill in fills} == {"10", "11"}
