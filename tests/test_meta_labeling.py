import numpy as np
import pandas as pd

from src.models.meta import build_meta_dataset, train_meta_model, META_BASE_FEATURES


def test_build_meta_dataset_labels_success_and_failure():
    n = 20
    df = pd.DataFrame({
        "close": [100.0] * n, "high": [100.0] * n, "low": [100.0] * n,
        "adx": [25.0] * n, "atr_pct": [0.01] * n,
        "volume_ratio": [1.0] * n, "volatility": [0.02] * n,
        "predicted_signal": [0] * n,
    })
    df.loc[0, "predicted_signal"] = 1
    df.loc[2, "high"] = 105.0  # TP пробит -> успешная сделка
    df.loc[5, "predicted_signal"] = 1
    df.loc[7, "low"] = 95.0  # SL пробит -> убыточная сделка

    meta_df = build_meta_dataset(df, transaction_cost=0.001)

    assert len(meta_df) == 2
    assert meta_df.iloc[0]["success"] == 1
    assert meta_df.iloc[1]["success"] == 0
    for feat in META_BASE_FEATURES:
        assert feat in meta_df.columns


def test_train_meta_model_returns_none_below_min_trades():
    meta_df = pd.DataFrame({
        "adx": [25.0, 26.0], "atr_pct": [0.01, 0.01],
        "volume_ratio": [1.0, 1.0], "volatility": [0.02, 0.02],
        "success": [1, 0],
    })
    model, feats = train_meta_model(meta_df, META_BASE_FEATURES, min_trades=30)
    assert model is None
    assert feats is None


def test_train_meta_model_trains_when_enough_trades():
    n = 40
    rng = np.random.default_rng(0)
    meta_df = pd.DataFrame({
        "adx": rng.uniform(10, 40, n), "atr_pct": rng.uniform(0.005, 0.02, n),
        "volume_ratio": rng.uniform(0.5, 2.0, n), "volatility": rng.uniform(0.01, 0.05, n),
        "success": rng.choice([0, 1], size=n),
    })
    model, feats = train_meta_model(meta_df, META_BASE_FEATURES, min_trades=30)
    assert model is not None
    assert feats == META_BASE_FEATURES