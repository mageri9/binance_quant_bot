import pytest
import pandas as pd
import numpy as np

from src.features.engineering import add_features
from src.labels.generator import generate_binary_labels, generate_triple_labels


def test_labels_correctness():
    # Создаем простой тестовый ряд цен закрытия
    # Индексы: 0    1    2    3    4    5    6
    # Цены:   100  101  102  105  103  104  106
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 105.0, 103.0, 104.0, 106.0],
            "volume": [100.0] * 7,
        }
    )

    # Тестируем бинарный лейбл с горизонтом = 2 свечи, без порога (threshold = 0)
    # Сравниваем цену t с ценой t+2:
    # t=0 (цена 100) -> t=2 (цена 102). Выросла? Да (1)
    # t=1 (цена 101) -> t=3 (цена 105). Выросла? Да (1)
    # t=2 (цена 102) -> t=4 (цена 103). Выросла? Да (1)
    # t=3 (цена 105) -> t=5 (цена 104). Выросла? Нет, упала (0)
    # t=4 (цена 103) -> t=6 (цена 106). Выросла? Да (1)
    # t=5, t=6 -> нет будущего на 2 шага вперед (NaN)

    binary_labels = generate_binary_labels(df, horizon=2, threshold=0.0)

    expected_binary = [1.0, 1.0, 1.0, 0.0, 1.0, np.nan, np.nan]
    pd.testing.assert_series_equal(
        binary_labels, pd.Series(expected_binary), check_names=False
    )


def test_triple_labels_correctness():
    # Цены: 100 -> через 1 свечу -> 102 (рост 2%), 100.5 (рост 0.5%), 97 (падение 3%)
    df = pd.DataFrame(
        {
            "close": [
                100.0,
                102.0,
                102.5,
                99.425,
            ],  # 99.425 - это падение от 102.5 на 3%
            "volume": [100.0] * 4,
        }
    )

    # Порог 1.5% (0.015)
    triple_labels = generate_triple_labels(df, horizon=1, threshold=0.015)

    # t=0 (100 -> 102): Рост 2% (> 1.5%) -> 1.0
    # t=1 (102 -> 102.5): Рост 0.49% (< 1.5%) -> 0.0 (Hold)
    # t=2 (102.5 -> 99.425): Падение 3% (< -1.5%) -> -1.0
    # t=3 -> NaN (нет будущего)

    expected_triple = [1.0, 0.0, -1.0, np.nan]
    pd.testing.assert_series_equal(
        triple_labels, pd.Series(expected_triple), check_names=False
    )


def test_feature_leakage_protection():
    """
    Математический Leakage-тест (защита от утечки данных).
    Проверяет, что изменение данных в будущем не влияет на расчет признаков в прошлом.
    """
    np.random.seed(42)
    n_rows = 60

    # Генерируем случайный ценовой ряд
    dummy_data = {
        "open_time": np.arange(1000, 1000 + n_rows),
        "open": np.random.uniform(100, 110, n_rows),
        "high": np.random.uniform(110, 120, n_rows),
        "low": np.random.uniform(90, 100, n_rows),
        "close": np.random.uniform(100, 110, n_rows),
        "volume": np.random.uniform(1000, 5000, n_rows),
    }
    df_original = pd.DataFrame(dummy_data)

    # 1. Считаем признаки для оригинальной истории
    df_feat_orig = add_features(df_original)

    # Выбираем контрольную точку во времени (например, 25-я свеча)
    control_idx = 25

    # Сохраняем рассчитанные признаки до момента control_idx включительно
    features_past_orig = df_feat_orig.iloc[: control_idx + 1].copy()

    # 2. Создаем модифицированную историю, где полностью искажаем будущее (все свечи после 25-й)
    df_modified = df_original.copy()
    df_modified.loc[control_idx + 1 :, "open"] = 9999.0
    df_modified.loc[control_idx + 1 :, "high"] = 9999.0
    df_modified.loc[control_idx + 1 :, "low"] = 1.0
    df_modified.loc[control_idx + 1 :, "close"] = 5555.0
    df_modified.loc[control_idx + 1 :, "volume"] = 999999.0

    # 3. Пересчитываем признаки на искаженной истории
    df_feat_mod = add_features(df_modified)
    features_past_mod = df_feat_mod.iloc[: control_idx + 1]

    # 4. Проверяем, изменились ли признаки прошлого из-за изменений в будущем
    feature_cols = [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility",
        "volume_ratio",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr",
        "adx",
    ]

    for col in feature_cols:
        # Убираем NaN (первые строки будут пустыми из-за окон расчета индикаторов)
        orig_series = features_past_orig[col].dropna()
        mod_series = features_past_mod[col].dropna()

        # Индексы непустых строк должны строго совпадать
        assert (orig_series.index == mod_series.index).all()
        # Значения должны остаться абсолютно идентичными!
        pd.testing.assert_series_equal(orig_series, mod_series, rtol=1e-12, atol=1e-12)

def test_label_leakage_removed_by_splitter_horizon():
    """
    Проверяет, что TimeSeriesWalkForwardSplitter с label_horizon
    отбрасывает из train_df строки, чья метка размечена по данным
    за пределами train-окна.
    """
    from src.models.backtest import TimeSeriesWalkForwardSplitter
    from src.labels.generator import generate_triple_labels

    n_rows = 200
    df = pd.DataFrame({
        "open_time": np.arange(n_rows),
        "close": np.linspace(100, 200, n_rows),
    })
    horizon = 5
    df["target_triple"] = generate_triple_labels(df, horizon=horizon, threshold=0.01)

    splitter = TimeSeriesWalkForwardSplitter(
        train_size=100, test_size=20, label_horizon=horizon,
    )

    for train_df, test_df, info in splitter.split(df):
        # Последняя строка train_df: метка должна быть посчитана по close,
        # который лежит строго внутри train-окна, а не в test-окне.
        last_train_idx = train_df.index[-1]
        # future_close для этой строки использовался close[last_train_idx + horizon]
        assert last_train_idx + horizon < info["test_start_idx"]
        # Обрезка реально произошла: train короче исходного train_size
        assert info["train_size"] == 100 - horizon
        break  # достаточно проверить первый фолд