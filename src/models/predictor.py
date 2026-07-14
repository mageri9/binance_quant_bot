import os
import pickle
import pandas as pd
from src.features.engineering import add_features


class Predictor:
    """
    Класс предсказателя. Загружает файл обученной модели и выдает сигналы
    по новым входящим свечам. Поддерживает как бинарные, так и тройные метки.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели по пути {model_path} не найден.")

        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)

        self.model = saved_data["model"]
        self.scaler = saved_data.get("scaler", None)

        # Защищенное получение признаков с жестким откатом на стандартные
        self.features = saved_data.get("features")
        if self.features is None:
            self.features = [
                "rsi",
                "macd",
                "macd_signal",
                "macd_hist",
                "volatility",
                "volume_ratio",
            ]

        # Защищенное получение названия целевой переменной
        self.target_col = saved_data.get("target_col")
        if self.target_col is None:
            self.target_col = "target_binary"

    def predict(self, df: pd.DataFrame) -> int | None:
        """
        Принимает DataFrame со свечами, рассчитывает по ним индикаторы
        и выдает сигнал на покупку (1), короткую продажу (-1) или флэт (0).
        """
        df_feats = add_features(df)

        # Нам нужен прогноз только для самой последней свечи
        latest_row = df_feats.iloc[-1]

        # Гарантируем, что список признаков не равен None
        features_to_check = self.features if self.features is not None else []

        # Проверяем, что признаки успешно рассчитались (нет NaN)
        if latest_row[features_to_check].isna().any():
            return None

        # Формируем строку признаков для модели
        X = pd.DataFrame([latest_row[features_to_check]])

        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Делаем предсказание [0, 1] или [0, 1, 2]
        pred = self.model.predict(X)[0]

        # Расшифровываем классы обратно
        target_col_str = (
            self.target_col if self.target_col is not None else "target_binary"
        )
        if target_col_str == "target_triple":
            # Маппинг: 0 -> -1 (Short), 1 -> 0 (Hold), 2 -> 1 (Long)
            if pred == 0:
                return -1
            elif pred == 1:
                return 0
            elif pred == 2:
                return 1

        return int(pred)