import numpy as np
import pandas as pd

from scripts.train import calculate_post_cost_targets


def test_post_cost_targets_apply_stops_timeout_and_commissions() -> None:
    commission_rate = 0.0004
    cases = [
        (105.0, 99.0, 104.0, 0.04, -0.02),
        (101.0, 95.0, 96.0, -0.02, 0.04),
        (101.0, 99.0, 101.0, 0.01, -0.01),
        (105.0, 95.0, 100.0, -0.02, -0.02),
    ]

    for high, low, close, expected_long, expected_short in cases:
        candles = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, high, high],
                "low": [100.0, low, low],
                "close": [100.0, close, close],
            }
        )

        long_returns, short_returns = calculate_post_cost_targets(
            candles, horizon=2, sl_pct=0.02, tp_pct=0.04,
            commission_rate=commission_rate,
        )

        assert long_returns.dtype == np.float32
        assert short_returns.dtype == np.float32
        assert np.isclose(
            long_returns[0],
            (1 + expected_long) * (1 - commission_rate) - (1 + commission_rate),
        )
        assert np.isclose(
            short_returns[0],
            (1 - commission_rate) - (1 - expected_short) * (1 + commission_rate),
        )
