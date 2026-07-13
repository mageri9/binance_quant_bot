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
        self.features = saved_data["features"]
        # Считываем имя целевой переменной, по умолчанию бинарная
        self.target_col = saved_data.get("target_col", "target_binary")

    def predict(self, df: pd.DataFrame) -> int | None:
        """
        Принимает DataFrame со свечами, рассчитывает по ним индикаторы
        и выдает сигнал на покупку (1), короткую продажу (-1) или флэт (0).
        """
        df_feats = add_features(df)

        # Нам нужен прогноз только для самой последней свечи
        latest_row = df_feats.iloc[-1]

        if latest_row[self.features].isna().any():
            return None

        # Формируем строку признаков для модели
        X = pd.DataFrame([latest_row[self.features]])

        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Делаем предсказание [0, 1] или [0, 1, 2]
        pred = self.model.predict(X)[0]

        # Расшифровываем классы обратно
        if self.target_col == "target_triple":
            # Маппинг: 0 -> -1 (Short), 1 -> 0 (Hold), 2 -> 1 (Long)
            if pred == 0:
                return -1
            elif pred == 1:
                return 0
            elif pred == 2:
                return 1

        return int(pred)