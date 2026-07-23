import numpy as np
import pandas as pd
import pytest

from src.execution.kernel import ExecutionCosts, ExecutionKernel
from src.models.side_comparison import SideComparisonConfig, build_side_net_returns, run_side_model_comparison


def _kernel():
    return ExecutionKernel(ExecutionCosts(commission_rate=0, slippage_rate=0, bid_ask_spread_rate=0, funding_rate_per_trade=0))


def test_side_net_returns_use_next_open_and_opposite_sides():
    df = pd.DataFrame({"open": np.arange(100, 110, dtype=float)})
    targets = build_side_net_returns(df, horizon=2, execution_kernel=_kernel())
    assert targets.loc[0, "long_net_return"] == pytest.approx((104 / 101) - 1)
    assert targets.loc[0, "short_net_return"] == pytest.approx((101 - 104) / 101)
    assert pd.isna(targets.loc[6, "long_net_return"])


def test_comparison_returns_trade_and_pf_deltas():
    rows = 90
    price = 100 + np.sin(np.arange(rows) / 3) + np.arange(rows) * 0.03
    df = pd.DataFrame({
        "open": price, "high": price + 0.5, "low": price - 0.5, "close": price,
        "feature": np.sin(np.arange(rows) / 3),
    })
    result = run_side_model_comparison(
        df, ["feature"], SideComparisonConfig(train_size=35, test_size=15, label_horizon=2), _kernel(),
    )
    assert result["policies"]["side_classifier"]["folds"] > 0
    assert "trade_change_pct" in result["policies"]["quantile_regression"]
    assert "profit_factor_delta" in result["policies"]["net_return_regression"]
