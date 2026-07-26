import pandas as pd
from src.strategy.features import add_features
from src.strategy.model import Predictor


class SignalGenerator:
    """Плагин стратегии: Свечи -> Индикаторы -> Модель -> Сигнал."""
    def __init__(self, model_path: str | None = None):
        self.predictor = Predictor(model_path) if model_path and os.path.exists(model_path) else None

    def generate(self, candles_df: pd.DataFrame) -> tuple[int, float]:
        if len(candles_df) < 30:
            return 0, 0.0

        df_feat = add_features(candles_df)
        if self.predictor is None:
            # Простой технический фолбэк, если модель еще не обучена
            latest = df_feat.iloc[-1]
            if latest["rsi"] < 30 and latest["macd_hist_pct"] > 0:
                return 1, 0.002
            elif latest["rsi"] > 70 and latest["macd_hist_pct"] < 0:
                return -1, 0.002
            return 0, 0.0

        return self.predictor.predict(df_feat)