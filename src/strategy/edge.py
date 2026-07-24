"""Expected-value gates for trade selection."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.strategy.signals import simulate_strategy


def apply_edge_threshold(
    df: pd.DataFrame,
    threshold: float,
    signal_col: str = "predicted_signal",
    expected_return_col: str = "predicted_expected_return",
) -> pd.DataFrame:
    """Turn signals without positive predicted net return into HOLD signals."""
    if expected_return_col not in df.columns:
        raise ValueError(f"Missing required expected-return column: {expected_return_col}")
    if signal_col not in df.columns:
        raise ValueError(f"Missing required signal column: {signal_col}")

    filtered = df.copy()
    keep = filtered[expected_return_col].gt(max(0.0, threshold)) & filtered[signal_col].ne(0)
    filtered[signal_col] = filtered[signal_col].where(keep, 0)
    return filtered


def sweep_edge_thresholds(
    df: pd.DataFrame,
    thresholds: Iterable[float],
    min_coverage: float,
    min_trades: int,
    simulate_kwargs: dict | None = None,
) -> tuple[float, list[dict]]:
    """Pick the most profitable sufficiently-covered expected-value gate.

    Candidates are ranked by expectancy, then profit factor, then coverage.
    The caller must pass the chronological calibration partition only; this
    function deliberately has no access to the final economic-test partition.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1].")
    if min_trades < 1:
        raise ValueError("min_trades must be at least one.")

    base_signals = int(df["predicted_signal"].ne(0).sum())
    if base_signals == 0:
        return 0.0, []

    kwargs = dict(simulate_kwargs or {})
    rows: list[dict] = []
    for threshold in sorted({float(value) for value in thresholds}):
        if threshold < 0:
            raise ValueError("minimum expected returns must be non-negative.")
        filtered = apply_edge_threshold(df, threshold)
        coverage = float(filtered["predicted_signal"].ne(0).sum() / base_signals)
        metrics = simulate_strategy(filtered, **kwargs)
        rows.append({
            "threshold": threshold,
            "coverage": coverage,
            "eligible": coverage >= min_coverage and metrics["total_trades"] >= min_trades,
            **metrics,
        })

    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return 0.0, rows
    best = max(
        eligible,
        key=lambda row: (row["expectancy"], row["profit_factor"], row["coverage"]),
    )
    return float(best["threshold"]), rows
