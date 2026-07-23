import pytest
from src.risk.engine import RiskEngine, RiskDecision


@pytest.mark.asyncio
async def test_risk_engine_circuit_breaker():
    engine = RiskEngine(consecutive_losses_limit=3)

    decision, amount, msg = await engine.validate_signal(
        symbol="BTC/USDT",
        side="buy",
        requested_amount=1.0,
        current_price=100.0,
        balance_free=1000.0,
        balance_total=1000.0,
        open_positions=[],
        closed_trades_last_24h=[],
        consecutive_losses=3,
    )

    assert decision == RiskDecision.DENY
    assert "Circuit Breaker" in msg


@pytest.mark.asyncio
async def test_risk_engine_max_daily_loss():
    engine = RiskEngine(max_daily_loss_pct=0.05)

    decision, amount, msg = await engine.validate_signal(
        symbol="BTC/USDT",
        side="buy",
        requested_amount=1.0,
        current_price=100.0,
        balance_free=1000.0,
        balance_total=1000.0,
        open_positions=[],
        closed_trades_last_24h=[{"pnl": -60.0}],
        consecutive_losses=0,
    )

    assert decision == RiskDecision.DENY
    assert "Превышен лимит потерь за 24ч" in msg


@pytest.mark.asyncio
async def test_risk_engine_max_open_positions():
    engine = RiskEngine(max_open_positions=2)

    open_positions = [
        {"symbol": "ETH/USDT", "side": "LONG", "amount": 1.0},
        {"symbol": "SOL/USDT", "side": "LONG", "amount": 1.0},
    ]

    decision, amount, msg = await engine.validate_signal(
        symbol="BTC/USDT",
        side="buy",
        requested_amount=1.0,
        current_price=100.0,
        balance_free=1000.0,
        balance_total=1000.0,
        open_positions=open_positions,
        closed_trades_last_24h=[],
        consecutive_losses=0,
    )

    assert decision == RiskDecision.DENY
    assert "Достигнут лимит открытых позиций" in msg


@pytest.mark.asyncio
async def test_risk_engine_allows_closing_order():
    engine = RiskEngine(max_open_positions=2)

    open_positions = [
        {"symbol": "BTC/USDT", "side": "LONG", "amount": 2.0},
        {"symbol": "ETH/USDT", "side": "LONG", "amount": 1.0},
    ]

    decision, amount, msg = await engine.validate_signal(
        symbol="BTC/USDT",
        side="sell",
        requested_amount=2.0,
        current_price=100.0,
        balance_free=1000.0,
        balance_total=1000.0,
        open_positions=open_positions,
        closed_trades_last_24h=[],
        consecutive_losses=0,
    )

    assert decision == RiskDecision.OPEN
    assert amount == 2.0
    assert "закрытия позиции одобрен" in msg


@pytest.mark.asyncio
async def test_risk_engine_reduces_allocation():
    engine = RiskEngine(max_allocation_pct=0.10)

    decision, amount, msg = await engine.validate_signal(
        symbol="BTC/USDT",
        side="buy",
        requested_amount=2.0,
        current_price=100.0,
        balance_free=1000.0,
        balance_total=1000.0,
        open_positions=[],
        closed_trades_last_24h=[],
        consecutive_losses=0,
    )

    assert decision == RiskDecision.REDUCE_SIZE
    assert amount == 1.0
    assert "Объем ордера снижен" in msg


@pytest.mark.asyncio
async def test_risk_engine_denies_same_side_add_to_open_position():
    engine = RiskEngine()

    decision, amount, msg = await engine.validate_signal(
        symbol="BTC/USDT",
        side="buy",
        requested_amount=1.0,
        current_price=100.0,
        balance_free=1000.0,
        balance_total=1000.0,
        open_positions=[{"symbol": "BTC/USDT", "side": "LONG", "amount": 2.0}],
        closed_trades_last_24h=[],
        consecutive_losses=0,
    )

    assert decision == RiskDecision.DENY
    assert amount == 0.0
    assert "Adding to an open position" in msg
