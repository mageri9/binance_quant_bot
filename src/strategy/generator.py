import os

import pandas as pd

from src.strategy.features import add_features
from src.strategy.model import Predictor


class SignalGenerator:
    """Transforms candles into a model signal, with a technical fallback."""

    def __init__(self, model_dir: str = "models/saved_models") -> None:
        self.model_dir = model_dir
        self.predictors: dict[tuple[str, str], Predictor] = {}

    def generate(
        self, candles_df: pd.DataFrame, symbol: str = "BTC/USDT", timeframe: str = "1h"
    ) -> tuple[int, float]:
        if len(candles_df) < 30:
            return 0, 0.0

        df_feat = add_features(candles_df)
        key = (symbol, timeframe)
        clean_sym = symbol.replace("/", "").replace(":", "")
        model_path = os.path.join(self.model_dir, f"lgbm_{clean_sym}_{timeframe}.pkl")

        if key not in self.predictors and os.path.exists(model_path):
            self.predictors[key] = Predictor(model_path)

        predictor = self.predictors.get(key)
        if predictor is not None:
            return predictor.predict(df_feat)

        latest = df_feat.iloc[-1]
        if latest["rsi"] < 30 and latest["macd_hist_pct"] > 0:
            return 1, 0.002
        if latest["rsi"] > 70 and latest["macd_hist_pct"] < 0:
            return -1, 0.002
        return 0, 0.0
