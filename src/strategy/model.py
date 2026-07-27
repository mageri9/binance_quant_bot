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
    def __init__(self, model_path: str = "models/active.pkl") -> None:
        self.model_path = model_path
        self.model_mtime = 0.0
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели {self.model_path} не найден.")

        model_mtime = os.path.getmtime(self.model_path)
        with open(self.model_path, "rb") as file:
            artifact = pickle.load(file)

        self.model = artifact["model"]
        self.features = artifact.get("features", [])
        self.min_expected_return = artifact.get("min_expected_return", 0.001)
        self.model_mtime = model_mtime

    def _reload_if_changed(self) -> None:
        if os.path.getmtime(self.model_path) != self.model_mtime:
            self._load_model()

    def predict(self, df_features: pd.DataFrame) -> tuple[int, float]:
        """Возвращает (сигнал: 1 | -1 | 0, expected_return: float)."""
        self._reload_if_changed()
        latest = df_features.iloc[[-1]]
        if latest[self.features].isna().any().any():
            return 0, 0.0

        X = latest[self.features]
        long_ret, short_ret = self.model.predict_returns(X)
        exp_long = float(long_ret[0])
        exp_short = float(short_ret[0])

        if exp_long > exp_short and exp_long >= self.min_expected_return:
            return 1, exp_long
        if exp_short > exp_long and exp_short >= self.min_expected_return:
            return -1, exp_short

        return 0, max(exp_long, exp_short)
