from decimal import Decimal

import pandas as pd
import pytest

from src.execution.kernel import ExecutionCosts
from src.execution.trade import TradePolicy, build_trade_targets, evaluate_trade


def _policy(**overrides):
    values = {
        "timeout_candles": 2,
        "sl_pct": Decimal("0.02"),
        "tp_pct": Decimal("0.03"),
        "costs": ExecutionCosts(
            commission_rate="0.001", slippage_rate="0.002",
            bid_ask_spread_rate="0.001", funding_rate_per_trade="0.0005",
        ),
    }
    values.update(overrides)
    return TradePolicy(**values)


def test_trade_example_is_the_next_open_order_with_full_economics():
    df = pd.DataFrame({
        "open_time": [1000, 2000, 3000, 4000],
        "open": [100, 110, 111, 112],
        "high": [101, 111, 114, 113],
        "low": [99, 109, 110, 111],
        "close": [100, 110, 111, 112],
    })
    policy = _policy()

    outcome = evaluate_trade(df, 0, "LONG", policy)

    assert outcome is not None
    assert outcome.spec.entry_time == 2000
    assert outcome.spec.entry_price == pytest.approx(110 * 1.0025)
    assert outcome.spec.sl == pytest.approx(110 * 0.98)
    assert outcome.spec.tp == pytest.approx(110 * 1.03)
    assert outcome.spec.timeout == 4000
    assert outcome.exit_reason == "take_profit"
    assert outcome.spec.commission > 0
    assert outcome.spec.slippage == pytest.approx(0.002)
    assert outcome.spec.funding > 0


def test_targets_expose_complete_order_and_share_the_policy():
    df = pd.DataFrame({
        "open_time": range(6), "open": [100] * 6, "close": [100] * 6,
        "high": [100, 100, 104, 100, 100, 100],
        "low": [100, 100, 99, 100, 100, 100],
    })
    targets = build_trade_targets(df, _policy())
    required = {
        "entry_price", "entry_time", "side", "sl", "tp", "timeout",
        "commission", "slippage", "funding", "exit_price", "exit_reason", "net_return",
    }
    assert required.issubset(targets.columns)
    assert targets.loc[0, "target_triple"] == 1

    changed = build_trade_targets(df, _policy(tp_pct=Decimal("0.10")))
    assert changed.loc[0, "target_triple"] == 0
