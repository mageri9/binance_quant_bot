import pandas as pd

from src.strategy.edge import apply_edge_threshold, sweep_edge_thresholds


def test_apply_edge_threshold_rejects_only_low_confidence_directional_signals():
    df = pd.DataFrame({
        "predicted_signal": [1, -1, 0],
        "predicted_confidence": [0.60, 0.80, 0.10],
    })

    result = apply_edge_threshold(df, 0.70)

    assert result["predicted_signal"].tolist() == [0, -1, 0]
    assert df["predicted_signal"].tolist() == [1, -1, 0]


def test_sweep_selects_higher_expectancy_gate_with_sufficient_coverage(monkeypatch):
    df = pd.DataFrame({
        "predicted_signal": [1, 1, 1],
        "predicted_confidence": [0.55, 0.70, 0.85],
    })

    def fake_simulate(filtered, **_kwargs):
        trade_count = int(filtered["predicted_signal"].ne(0).sum())
        expectancy = {3: 0.01, 2: 0.03, 1: 0.02}[trade_count]
        return {
            "total_trades": trade_count,
            "expectancy": expectancy,
            "profit_factor": float(trade_count),
        }

    monkeypatch.setattr("src.strategy.edge.simulate_strategy", fake_simulate)

    threshold, rows = sweep_edge_thresholds(
        df, thresholds=[0.55, 0.70, 0.85], min_coverage=0.5, min_trades=1,
    )

    assert threshold == 0.70
    assert [row["coverage"] for row in rows] == [1.0, 2 / 3, 1 / 3]
    assert rows[-1]["eligible"] is False
