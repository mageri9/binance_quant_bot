import numpy as np
import pandas as pd
import pytest

import scripts.train as train
from scripts.train import FEATURE_COLS, calculate_post_cost_targets


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


@pytest.mark.parametrize(
    ("prediction_count", "oos_return"),
    [(30, -0.01), (10, 0.01)],
)
def test_quality_gate_does_not_overwrite_model_when_oos_is_unacceptable(
    monkeypatch, tmp_path, prediction_count: int, oos_return: float
) -> None:
    class FakeModel:
        def __init__(self, **kwargs) -> None:
            self.fit_calls = 0

        def fit(self, X, y_long, y_short) -> None:
            self.fit_calls += 1

        def predict_returns(self, X):
            predictions = np.zeros(len(X))
            predictions[:prediction_count] = 0.01
            return predictions, np.zeros(len(X))

    monkeypatch.setattr(train, "EconomicReturnRegressor", FakeModel)
    monkeypatch.chdir(tmp_path)

    df_clean = pd.DataFrame({column: np.arange(100, dtype=float) for column in FEATURE_COLS})
    df_clean["long_target"] = 0.01
    df_clean["short_target"] = 0.01
    df_clean.loc[70:, "long_target"] = oos_return

    model_path = tmp_path / "models" / "saved_models" / "lgbm_BTCUSDT_1h.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"existing approved model")

    settings = type("Settings", (), {"MIN_EXPECTED_RETURN": 0.001})()
    saved = train.train_and_save_model(df_clean, settings, "BTC/USDT", "1h", horizon=5)

    assert saved is False
    assert model_path.read_bytes() == b"existing approved model"


def test_oos_uses_the_same_single_direction_rule_as_predictor() -> None:
    class FakeModel:
        def predict_returns(self, X):
            return np.array([0.01, 0.02, 0.01]), np.array([0.02, 0.01, 0.01])

    test_df = pd.DataFrame({column: [0.0, 0.0, 0.0] for column in FEATURE_COLS})
    test_df["long_target"] = [0.50, 0.30, 0.90]
    test_df["short_target"] = [0.10, 0.40, 0.80]

    n_trades, mean_return = train.evaluate_oos(FakeModel(), test_df, 0.001)

    assert n_trades == 2
    assert mean_return == pytest.approx(0.20)


def test_quality_gate_purges_overlapping_train_targets_before_refit(monkeypatch) -> None:
    fit_sizes: list[int] = []
    published_artifacts = []

    class FakeModel:
        def __init__(self, **kwargs) -> None:
            pass

        def fit(self, X, y_long, y_short) -> None:
            fit_sizes.append(len(X))

        def predict_returns(self, X):
            return np.full(len(X), 0.01), np.zeros(len(X))

    monkeypatch.setattr(train, "EconomicReturnRegressor", FakeModel)
    monkeypatch.setattr(
        train, "save_model_artifact", lambda artifact, path: published_artifacts.append((artifact, path))
    )
    df_clean = pd.DataFrame({column: np.arange(100, dtype=float) for column in FEATURE_COLS})
    df_clean["long_target"] = 0.01
    df_clean["short_target"] = -0.01
    settings = type("Settings", (), {"MIN_EXPECTED_RETURN": 0.001})()

    saved = train.train_and_save_model(df_clean, settings, "BTC/USDT", "1h", horizon=5)

    assert saved is True
    assert fit_sizes == [65, 100]
    assert published_artifacts[0][1] == "models/saved_models/lgbm_BTCUSDT_1h.pkl"
