import os
import pickle
import pandas as pd
from src.features.engineering import add_features


class Predictor:
    """
    Класс предсказателя. Загружает упакованный артефакт модели и выдает сигналы
    по новым входящим свечам, а также предоставляет параметры калибровки рисков.
    """

    def __init__(self, model_path: str, confidence_threshold: float | None = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели по пути {model_path} не найден.")

        if confidence_threshold is None:
            from src.core.config import get_settings

            confidence_threshold = get_settings().PREDICTION_CONFIDENCE_THRESHOLD
        self.confidence_threshold = confidence_threshold

        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)

        self.model = saved_data["model"]
        self.scaler = saved_data.get("scaler", None)

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

        self.target_col = saved_data.get("target_col")
        if self.target_col is None:
            self.target_col = "target_binary"

        # Извлекаем новые MLOps-метаданные артефакта
        self.model_id = saved_data.get("model_id", "legacy_model")
        self.dataset_version = saved_data.get("dataset_version", "unknown")
        self.git_sha = saved_data.get("git_sha", "unknown")
        self.features_hash = saved_data.get("features_hash", "unknown")
        self.calibration = saved_data.get(
            "calibration", {"sl_pct": 0.02, "tp_pct": 0.04}
        )

    def predict(self, df: pd.DataFrame) -> int | None:
        """
        Сохраняем старый метод полностью совместимым.
        """
        signal, _ = self.predict_detailed(df)
        return signal

    def predict_detailed(self, df: pd.DataFrame) -> tuple[int | None, dict | None]:
        """
        Возвращает: (signal, probabilities_dict).
        Выдает класс прогноза и словарь вероятностей классов для MLOps-логирования.
        """
        df_feats = add_features(df)

        latest_row = df_feats.iloc[-1]
        features_to_check = self.features if self.features is not None else []

        if latest_row[features_to_check].isna().any():
            return None, None

        X = pd.DataFrame([latest_row[features_to_check]])

        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Вычисляем сырые вероятности классов
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[
            0
        ].tolist()  # Массив [p0, p1, p2] или [p0, p1]

        target_col_str = (
            self.target_col if self.target_col is not None else "target_binary"
        )

        prob_dict = {}
        if target_col_str == "target_triple":
            # Тройная разметка (классы 0 -> Short (-1), 1 -> Hold (0), 2 -> Long (1))
            prob_dict = {
                "prob_short": prob[0],
                "prob_hold": prob[1],
                "prob_long": prob[2],
            }
            if pred == 0:
                signal = -1
            elif pred == 1:
                signal = 0
            elif pred == 2:
                signal = 1
        else:
            # Бинарная разметка (классы 0 -> Hold/Short, 1 -> Long)
            prob_dict = {"prob_short": 0.0, "prob_hold": prob[0], "prob_long": prob[1]}
            signal = int(pred)

        if signal == 1 and prob_dict["prob_long"] < self.confidence_threshold:
            signal = 0
        elif signal == -1 and prob_dict["prob_short"] < self.confidence_threshold:
            signal = 0

        return signal, prob_dict