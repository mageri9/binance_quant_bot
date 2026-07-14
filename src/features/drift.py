import pandas as pd
from scipy.stats import ks_2samp
from loguru import logger


class ConceptDriftDetector:
    """
    Детектор концептуального дрейфа признаков на основе статистического критерия Колмогорова-Смирнова.
    """

    @staticmethod
    def detect_drift(
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: list[str],
        threshold: float = 0.05,
    ) -> dict:
        """
        Сравнивает распределение признаков между референсным (архивным) датасетом и новым.
        Возвращает детальный отчет о смещениях p-value и общий флаг обнаружения дрейфа.
        """
        results = {}
        drift_detected = False

        for col in features:
            if col not in reference_df.columns or col not in current_df.columns:
                continue

            ref_data = reference_df[col].dropna()
            cur_data = current_df[col].dropna()

            # Исключаем расчет на пустых или слишком малых выборках
            if len(ref_data) < 15 or len(cur_data) < 15:
                continue

            # Двусторонний тест Колмогорова-Смирнова
            stat, p_value = ks_2samp(ref_data, cur_data)

            # Если p-value ниже альфа-порога, гипотеза о равенстве распределений отвергается
            is_drifted = p_value < threshold
            results[col] = {
                "stat": float(stat),
                "p_value": float(p_value),
                "drift": is_drifted,
            }

            if is_drifted:
                drift_detected = True
                logger.warning(
                    f"[Drift SRE] Обнаружен статистический дрейф признака '{col}': "
                    f"p-value={p_value:.5f} (< {threshold})"
                )

        return {"drift_detected": drift_detected, "metrics": results}