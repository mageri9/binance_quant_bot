import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


class EconomicReturnRegressor:
    """Оценка доходности направления LONG и SHORT через LightGBM."""
    def __init__(self, **kwargs):
        self.long_model = LGBMRegressor(**kwargs)
        self.short_model = LGBMRegressor(**kwargs)

    def fit(self, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series):
        self.long_model.fit(X, y_long)
        self.short_model.fit(X, y_short)

    def predict_returns(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.long_model.predict(X), self.short_model.predict(X)


class Predictor:
    """Предиктор для сохраненного артефакта модели."""
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели {model_path} не найден.")
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        self.model = artifact["model"]
        self.features = artifact.get("features", [])
        self.min_expected_return = artifact.get("min_expected_return", 0.001)

    def predict(self, df_features: pd.DataFrame) -> tuple[int, float]:
        """Возвращает (сигнал: 1 | -1 | 0, expected_return: float)."""
        latest = df_features.iloc[[-1]]
        if latest[self.features].isna().any().any():
            return 0, 0.0

        X = latest[self.features]
        long_ret, short_ret = self.model.predict_returns(X)
        exp_long = float(long_ret[0])
        exp_short = float(short_ret[0])

        if exp_long > exp_short and exp_long >= self.min_expected_return:
            return 1, exp_long
        elif exp_short > exp_long and exp_short >= self.min_expected_return:
            return -1, exp_short

        return 0, max(exp_long, exp_short)