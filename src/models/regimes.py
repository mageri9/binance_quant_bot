"""Leakage-safe soft market-regime ensemble helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


REGIME_NAMES = ("bear", "range", "bull")


def soft_regime_weights(df: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    """Return causal bear/range/bull memberships for every input row.

    The trend score is normalised by trailing realized volatility.  It has no
    fitted state, so exactly the same calculation is valid in WFO and live use.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if "close" in df:
        close = pd.to_numeric(df["close"], errors="coerce")
        returns = close.pct_change()
        trend = close.pct_change(20)
        volatility = returns.rolling(20, min_periods=5).std()
        score = trend / (volatility * np.sqrt(20) + 1e-8)
    elif "rsi" in df:
        # Feature-only model inputs do not contain price. RSI preserves a
        # causal, scale-free trend proxy at inference time.
        score = (pd.to_numeric(df["rsi"], errors="coerce") - 50.0) / 10.0
    else:
        raise ValueError("Soft regime ensemble requires close or rsi.")
    score = score.clip(-5.0, 5.0).fillna(0.0)

    # A small prior keeps the ambiguous centre genuinely range-dominant while
    # retaining continuous transitions into directional regimes.
    logits = np.column_stack((-score, 0.5 - np.abs(score), score)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return pd.DataFrame(weights, index=df.index, columns=REGIME_NAMES)


@dataclass
class SoftRegimeEnsemble:
    """Three sample-weighted classifiers combined using current memberships."""

    models: list[LGBMClassifier]
    classes_: np.ndarray
    temperature: float = 1.0
    regime_names: tuple[str, ...] = REGIME_NAMES

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        weights = soft_regime_weights(X, self.temperature).loc[:, self.regime_names].to_numpy()
        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
        for column, model in enumerate(self.models):
            model_proba = model.predict_proba(X)
            aligned = np.zeros_like(probabilities)
            for source_idx, class_label in enumerate(model.classes_):
                target_idx = int(np.where(self.classes_ == class_label)[0][0])
                aligned[:, target_idx] = model_proba[:, source_idx]
            probabilities += aligned * weights[:, [column]]
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def fit_soft_regime_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    model_kwargs: dict,
    temperature: float = 1.0,
) -> SoftRegimeEnsemble:
    """Fit specialists with fractional regime membership as sample weights."""
    classes = np.sort(np.asarray(pd.unique(y)))
    if len(classes) < 2:
        raise ValueError("Soft regime ensemble requires at least two target classes.")

    weights = soft_regime_weights(X, temperature).loc[:, REGIME_NAMES]
    models = []
    for regime in REGIME_NAMES:
        model = LGBMClassifier(**model_kwargs)
        model.fit(X, y, sample_weight=weights[regime].to_numpy())
        models.append(model)
    return SoftRegimeEnsemble(models=models, classes_=classes, temperature=temperature)
