"""Economic acceptance criteria shared by training and promotion."""

from __future__ import annotations

from math import isfinite
from typing import Mapping


def economic_quality_failure(
    metrics: Mapping[str, float] | None,
    *,
    min_trades: int,
    champion_metrics: Mapping[str, float] | None = None,
) -> str | None:
    """Return a rejection reason, or ``None`` when OOS economics are acceptable."""
    if not metrics:
        return "economic OOS metrics are unavailable"
    if int(metrics.get("total_trades", 0)) < min_trades:
        return f"only {int(metrics.get('total_trades', 0))} OOS trades; need {min_trades}"
    expectancy = float(metrics.get("expectancy", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    if not isfinite(expectancy) or expectancy <= 0:
        return "non-positive expected return"
    if not isfinite(profit_factor) or profit_factor <= 1.0:
        return "profit factor is not above 1.0"
    if champion_metrics:
        challenger_sharpe = float(metrics.get("sharpe_ci_low", metrics.get("sharpe_ratio", 0.0)))
        champion_sharpe = float(champion_metrics.get("sharpe_ci_high", champion_metrics.get("sharpe_ratio", 0.0)))
        if challenger_sharpe <= champion_sharpe:
            return f"conservative Sharpe ({challenger_sharpe:.3f}) does not exceed champion Sharpe ({champion_sharpe:.3f})"
    return None
