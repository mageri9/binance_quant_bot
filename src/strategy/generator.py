import os

import pandas as pd
from src.strategy.features import add_features
from src.strategy.model import Predictor


class SignalGenerator:
    """Плагин стратегии: Свечи -> Индикаторы -> Модель -> Сигнал."""
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or "models/active.pkl"
        self.predictor = Predictor(self.model_path) if os.path.exists(self.model_path) else None

    def generate(self, candles_df: pd.DataFrame) -> tuple[int, float]:
        if len(candles_df) < 30:
            return 0, 0.0

        df_feat = add_features(candles_df)
        if self.predictor is None:
            if os.path.exists(self.model_path):
                self.predictor = Predictor(self.model_path)
                return self.predictor.predict(df_feat)

            # Простой технический фолбэк, если модель еще не обучена
            latest = df_feat.iloc[-1]
            if latest["rsi"] < 30 and latest["macd_hist_pct"] > 0:
                return 1, 0.002
            elif latest["rsi"] > 70 and latest["macd_hist_pct"] < 0:
                return -1, 0.002
            return 0, 0.0

        return self.predictor.predict(df_feat)
