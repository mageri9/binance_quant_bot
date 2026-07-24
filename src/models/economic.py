"""Side-specific post-cost return regression used by production predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ECONOMIC_TARGETS = ("long_net_return", "short_net_return")


class EconomicReturnRegressor:
    """Estimate the realized net return of the executable LONG and SHORT trades."""

    def __init__(self, **model_kwargs):
        self.long_model = LGBMRegressor(**model_kwargs)
        self.short_model = LGBMRegressor(**model_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "EconomicReturnRegressor":
        missing = set(ECONOMIC_TARGETS).difference(y.columns)
        if missing:
            raise ValueError(f"Missing economic targets: {sorted(missing)}")
        self.long_model.fit(X, y["long_net_return"])
        self.short_model.fit(X, y["short_net_return"])
        return self

    def predict_returns(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.long_model.predict(X), self.short_model.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        long_return, short_return = self.predict_returns(X)
        return np.maximum(long_return, short_return)

    def signals(self, X: pd.DataFrame, min_expected_return: float = 0.0) -> np.ndarray:
        long_return, short_return = self.predict_returns(X)
        best = np.maximum(long_return, short_return)
        minimum_ev = max(0.0, min_expected_return)
        return np.where(best <= minimum_ev, 0, np.where(long_return >= short_return, 1, -1))
