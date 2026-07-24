import os
import pickle
import pandas as pd
from src.features.engineering import add_features


class Predictor:
    """
    Класс предсказателя. Загружает упакованный артефакт модели и выдает сигналы
    по новым входящим свечам, а также предоставляет параметры калибровки рисков.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float | None = None,
        meta_threshold: float | None = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели по пути {model_path} не найден.")

        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)

        if confidence_threshold is None:
            from src.core.config import get_settings
            confidence_threshold = saved_data.get(
                "edge_threshold", get_settings().PREDICTION_CONFIDENCE_THRESHOLD,
            )
        self.confidence_threshold = confidence_threshold

        if meta_threshold is None:
            from src.core.config import get_settings
            meta_threshold = get_settings().META_LABELING_THRESHOLD
        self.meta_threshold = meta_threshold

        self.model = saved_data["model"]
        self.scaler = saved_data.get("scaler", None)
        self.meta_model = saved_data.get("meta_model")
        self.meta_features = saved_data.get("meta_features")
        self.regime_drift_pvalue_at_train = saved_data.get("regime_drift_pvalue_at_train")

        self.features = saved_data.get("features")
        if self.features is None:
            self.features = ["rsi", "macd", "macd_signal", "macd_hist", "volatility", "volume_ratio"]

        self.target_col = saved_data.get("target_col")
        if self.target_col is None:
            self.target_col = "target_binary"

        self.model_id = saved_data.get("model_id", "legacy_model")
        self.dataset_version = saved_data.get("dataset_version", "unknown")
        self.git_sha = saved_data.get("git_sha", "unknown")
        self.features_hash = saved_data.get("features_hash", "unknown")
        self.calibration = saved_data.get("calibration", {"sl_pct": 0.02, "tp_pct": 0.04})
        self.model_type = saved_data.get("model_type", "classification")
        from src.core.config import get_settings
        self.min_expected_return = saved_data.get(
            "min_expected_return", get_settings().MIN_EXPECTED_RETURN,
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

        if self.model_type == "economic_return_regression":
            long_return, short_return = self.model.predict_returns(X)
            expected_long = float(long_return[0])
            expected_short = float(short_return[0])
            expected_return = max(expected_long, expected_short)
            signal = 0 if expected_return <= self.min_expected_return else (
                1 if expected_long >= expected_short else -1
            )
            return signal, {
                "expected_long_return": expected_long,
                "expected_short_return": expected_short,
                "expected_return": expected_return,
                "min_expected_return": self.min_expected_return,
            }

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

        if signal != 0 and self.meta_model is not None and self.meta_features:
            meta_row = {}
            meta_ok = True
            for feat in self.meta_features:
                if feat == "predicted_signal":
                    meta_row[feat] = signal
                elif feat == "predicted_confidence":
                    meta_row[feat] = max(prob)
                elif feat == "regime_drift_pvalue":
                    if self.regime_drift_pvalue_at_train is None:
                        meta_ok = False
                        break
                    meta_row[feat] = self.regime_drift_pvalue_at_train
                elif feat in latest_row.index and pd.notna(latest_row[feat]):
                    meta_row[feat] = latest_row[feat]
                else:
                    meta_ok = False
                    break

            if meta_ok:
                meta_X = pd.DataFrame([meta_row])[self.meta_features]
                meta_proba = self.meta_model.predict_proba(meta_X)[0]
                classes = list(self.meta_model.classes_)
                success_idx = classes.index(1) if 1 in classes else 1
                meta_success_prob = meta_proba[success_idx]

                if meta_success_prob < self.meta_threshold:
                    signal = 0

        return signal, prob_dict
