"""Leakage-safe walk-forward feature ablations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.features.engineering import OHLCV_FEATURE_GROUPS, TECHNICAL_FEATURES
from src.labels.generator import MAX_ADAPTIVE_HORIZON_CANDLES
from src.models.backtest import TimeSeriesWalkForwardSplitter


def ohlcv_ablation_feature_sets() -> dict[str, list[str]]:
    """Return technical baseline, one-family additions, and full OHLCV set."""
    feature_sets = {"technical_baseline": list(TECHNICAL_FEATURES)}
    for name, features in OHLCV_FEATURE_GROUPS.items():
        feature_sets[f"technical_plus_{name}"] = TECHNICAL_FEATURES + features
    feature_sets["technical_plus_all_ohlcv"] = TECHNICAL_FEATURES + [
        feature for features in OHLCV_FEATURE_GROUPS.values() for feature in features
    ]
    return feature_sets


def run_ohlcv_ablation(
    df: pd.DataFrame,
    target_col: str,
    train_size: int = 1000,
    test_size: int = 200,
) -> dict:
    """Score each feature family on identical chronological OOS folds."""
    if target_col not in df:
        raise ValueError(f"Target column {target_col!r} is missing from the dataset.")

    is_multiclass = target_col == "target_triple"
    average = "macro" if is_multiclass else "binary"
    splitter = TimeSeriesWalkForwardSplitter(
        train_size=train_size,
        test_size=test_size,
        label_horizon=MAX_ADAPTIVE_HORIZON_CANDLES,
    )
    results = []
    baseline_f1 = None

    for name, features in ohlcv_ablation_feature_sets().items():
        missing = sorted(set(features) - set(df.columns))
        if missing:
            raise ValueError(f"Feature set {name!r} is missing columns: {missing}")
        prepared = df.dropna(subset=features + [target_col]).reset_index(drop=True)
        y_true: list[int] = []
        y_pred: list[int] = []

        for train_df, test_df, _ in splitter.split(prepared):
            train_y = train_df[target_col]
            test_y = test_df[target_col]
            if is_multiclass:
                train_y = train_y.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
                test_y = test_y.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
            else:
                train_y = train_y.astype(int)
                test_y = test_y.astype(int)

            scaler = StandardScaler()
            x_train = scaler.fit_transform(train_df[features])
            x_test = scaler.transform(test_df[features])
            model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
            model.fit(x_train, train_y)
            y_true.extend(test_y.tolist())
            y_pred.extend(model.predict(x_test).tolist())

        if not y_true:
            raise ValueError("No walk-forward folds were created; increase dataset size or reduce window sizes.")
        f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        f1_delta = None if baseline_f1 is None else f1 - baseline_f1
        if name == "technical_baseline":
            baseline_f1 = f1
        results.append({
            "name": name,
            "features": features,
            "f1": f1,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "oos_samples": len(y_true),
            "f1_delta_vs_baseline": f1_delta,
        })

    return {
        "target": target_col,
        "train_size": train_size,
        "test_size": test_size,
        "results": results,
    }
