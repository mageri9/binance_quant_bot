import numpy as np
import pandas as pd

import pytest

from src.models.meta import (
    META_BASE_FEATURES,
    PRIMARY_OOF_FOLD_COLUMN,
    PRIMARY_OOF_ROW_COLUMN,
    PRIMARY_TRAIN_END_COLUMN,
    build_cross_fitted_meta_dataset,
    build_meta_dataset,
    train_meta_model,
)


def test_build_meta_dataset_uses_realized_net_return_as_target():
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
    assert meta_df.iloc[0]["future_net_return"] > 0
    assert meta_df.iloc[1]["future_net_return"] < 0
    for feat in META_BASE_FEATURES:
        assert feat in meta_df.columns


def test_cross_fitted_meta_dataset_rejects_unprovenanced_primary_predictions():
    df = pd.DataFrame({
        "close": [100.0], "high": [100.0], "low": [100.0],
        "adx": [25.0], "atr_pct": [0.01], "volume_ratio": [1.0],
        "volatility": [0.02], "predicted_signal": [1],
    })

    with pytest.raises(ValueError, match="cross-fitted provenance"):
        build_cross_fitted_meta_dataset(df)


def test_cross_fitted_meta_dataset_rejects_in_sample_primary_prediction():
    df = pd.DataFrame({
        "close": [100.0, 100.0], "high": [100.0, 105.0], "low": [100.0, 100.0],
        "adx": [25.0, 25.0], "atr_pct": [0.01, 0.01], "volume_ratio": [1.0, 1.0],
        "volatility": [0.02, 0.02], "predicted_signal": [1, 0],
        PRIMARY_OOF_FOLD_COLUMN: [0, 0], PRIMARY_TRAIN_END_COLUMN: [0, 0],
        PRIMARY_OOF_ROW_COLUMN: [0, 1],
    })

    with pytest.raises(ValueError, match="non-cross-fitted"):
        build_cross_fitted_meta_dataset(df)


def test_train_meta_model_returns_none_below_min_trades():
    meta_df = pd.DataFrame({
        "adx": [25.0, 26.0], "atr_pct": [0.01, 0.01],
        "volume_ratio": [1.0, 1.0], "volatility": [0.02, 0.02],
        "future_net_return": [0.01, -0.01],
    })
    model, feats, metrics = train_meta_model(meta_df, META_BASE_FEATURES, min_trades=30)
    assert model is None
    assert feats is None
    assert metrics["rejected_reason"] == "insufficient_trades"


def test_train_meta_model_trains_when_enough_trades():
    n = 60
    rng = np.random.default_rng(1)
    success = np.tile([1, 0], n // 2)  # чередование 1,0,1,0... вместо блоков —
                                          # гарантирует оба класса в любом хронологическом срезе
    adx = np.where(
        success == 1,
        rng.uniform(35, 45, n),
        rng.uniform(5, 15, n),
    )
    meta_df = pd.DataFrame({
        "adx": adx, "atr_pct": rng.uniform(0.005, 0.02, n),
        "volume_ratio": rng.uniform(0.5, 2.0, n), "volatility": rng.uniform(0.01, 0.05, n),
        "future_net_return": np.where(success == 1, 0.02, -0.01),
    })
    model, feats, metrics = train_meta_model(meta_df, META_BASE_FEATURES, min_trades=30)
    assert model is not None
    assert feats == META_BASE_FEATURES
    assert metrics["rejected_reason"] is None


def test_train_meta_model_rejects_when_no_lift_over_baseline():
        n = 60
        meta_df = pd.DataFrame(
            {
                "adx": [25.0] * n,
                "atr_pct": [0.01] * n,
                "volume_ratio": [1.0] * n,
                "volatility": [0.02] * n,
                "future_net_return": (
                    [1, 0] * (n // 2)
                ),  # признаки одинаковые для success=0/1 -> модель не может отличить
            }
        )
        model, feats, metrics = train_meta_model(
            meta_df, META_BASE_FEATURES, min_trades=30
        )
        assert model is None
        assert metrics["rejected_reason"] in (
            "no_expectancy_lift_over_baseline",
        )

def test_train_meta_model_accepts_when_features_separate_classes():
    n = 60
    rng = np.random.default_rng(1)
    success = np.tile([1, 0], n // 2)
    adx = np.where(
        success == 1,
        rng.uniform(35, 45, n),
        rng.uniform(5, 15, n),
    )
    meta_df = pd.DataFrame({
        "adx": adx, "atr_pct": rng.uniform(0.005, 0.02, n),
        "volume_ratio": rng.uniform(0.5, 2.0, n), "volatility": rng.uniform(0.01, 0.05, n),
        "future_net_return": np.where(success == 1, 0.02, -0.01),
    })
    model, feats, metrics = train_meta_model(meta_df, META_BASE_FEATURES, min_trades=30)
    assert model is not None
    assert metrics["rejected_reason"] is None
    assert metrics["approved_expectancy"] > metrics["baseline_expectancy"]
