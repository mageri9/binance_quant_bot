import os
import pickle
import pandas as pd
from src.features.engineering import add_features


class Predictor:
    """
    Класс предсказателя. Загружает файл обученной модели и выдает сигналы
    по новым входящим свечам.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели по пути {model_path} не найден.")

        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)

        self.model = saved_data["model"]
        self.scaler = saved_data.get("scaler", None)
        self.features = saved_data["features"]

    def predict(self, df: pd.DataFrame) -> int | None:
        """
        Принимает DataFrame со свечами, рассчитывает по ним индикаторы
        и выдает сигнал на покупку (1) или продажу/флэт (0).
        """
        # 1. Рассчитываем признаки
        df_feats = add_features(df)

        # Нам нужен прогноз только для самой последней свечи
        latest_row = df_feats.iloc[-1]

        # Проверяем, что признаки успешно рассчитались (нет NaN)
        if latest_row[self.features].isna().any():
            return None

        # Формируем строку признаков для модели
        X = pd.DataFrame([latest_row[self.features]])

        # Масштабируем признаки, если это необходимо (для моделей типа регрессии)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Делаем предсказание
        pred = self.model.predict(X)[0]
        return int(pred)