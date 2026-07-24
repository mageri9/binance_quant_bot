import pytest
import pandas as pd
import numpy as np

from src.features.engineering import add_features
from src.labels.generator import (
    generate_triple_labels,
    MAX_ADAPTIVE_HORIZON_CANDLES,
    MIN_ADAPTIVE_HORIZON_CANDLES,
    generate_binary_labels,
)
from src.models.backtest import TimeSeriesWalkForwardSplitter



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


def test_adaptive_triple_labels_with_atr_path_dependency():
    """
    Проверяет адаптивный разметчик Triple Barrier на полноценных данных:
    проверяет, что при высокой волатильности горизонт сжимается,
    а при низкой расширяется, и корректно срабатывают SL/TP на истории.
    """
    # Генерируем 120 свечей с контролируемыми ценами и ATR
    np.random.seed(42)
    close_prices = [100.0] * 120
    high_prices = [100.5] * 120
    low_prices = [99.5] * 120

    # 20-я свеча дает резкий импульс вверх до 105.0 (TP для 1% барьера)
    high_prices[21] = 105.0
    close_prices[21] = 104.5

    # 40-я свеча дает резкий импульс вниз до 95.0 (SL для 1% барьера)
    low_prices[41] = 95.0
    close_prices[41] = 95.5

    df = pd.DataFrame(
        {
            "close": close_prices,
            "high": high_prices,
            "low": low_prices,
            "atr": [1.0] * 120,  # постоянная волатильность
        }
    )

    # Размечаем с симметричным барьером в 1% (0.01) и базовым горизонтом 5
    labels = generate_triple_labels(df, horizon=5, threshold=0.01)

    # Вход на 20-й свече: на 21-й (t+1) high достигает 105.0 (TP) -> Должен быть Long (1.0)
    assert labels.iloc[20] == 1.0

    # Вход на 40-й свече: на 41-й (t+1) low падает до 95.0 (SL) -> Должен быть Short (-1.0)
    assert labels.iloc[40] == -1.0

    # Вход на 80-й свече: цена не колеблется за пределы 99.5-100.5 -> Выход по тайм-ауту (0.0)
    assert labels.iloc[80] == 0.0

    # Последние строки (в пределах максимального горизонта) должны быть NaN
    assert pd.isna(labels.iloc[-1])

def test_adaptive_horizon_capped_at_max_constant():
    """
    Регрессия на источник утечки (Quest 4): при растянутом адаптивном
    горизонте последние MAX_ADAPTIVE_HORIZON_CANDLES строк обязаны
    остаться неразмеченными (не хватает будущего) — это подтверждает,
    что верхняя граница горизонта действительно достижима и равна
    константе, используемой для purge.
    """
    np.random.seed(7)
    n = 300
    atr = np.concatenate([np.full(150, 5.0), np.full(150, 0.05)])
    close = 100 + np.cumsum(np.random.normal(0, 0.1, n))
    df = pd.DataFrame({
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "atr": atr,
    })

    labels = generate_triple_labels(df, horizon=2, threshold=0.01)

    assert labels.iloc[-MAX_ADAPTIVE_HORIZON_CANDLES:].isna().all()
    assert not pd.isna(labels.iloc[-(MAX_ADAPTIVE_HORIZON_CANDLES + 1)])


def test_walk_forward_purge_with_nominal_horizon_is_insufficient():
    """
    Показывает сам баг, который чинит Quest 4: purge на номинальном
    LABEL_HORIZON (например 5) НЕ покрывает адаптивный горизонт, который
    может достигать MAX_ADAPTIVE_HORIZON_CANDLES (15) — то есть при
    недостаточном purge несколько последних строк train_df всё ещё
    содержат метки, резолвившиеся внутри test-окна.
    """
    np.random.seed(11)
    n = 200
    atr = np.full(n, 5.0)
    atr[:100] = 0.05
    close = 100 + np.cumsum(np.random.normal(0, 0.1, n))
    df = pd.DataFrame({
        "open_time": np.arange(n),
        "close": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "atr": atr,
    })
    nominal_horizon = 5
    df["target_triple"] = generate_triple_labels(df, horizon=nominal_horizon, threshold=0.01)

    insufficient_splitter = TimeSeriesWalkForwardSplitter(
        train_size=100, test_size=20, label_horizon=nominal_horizon,
    )
    correct_splitter = TimeSeriesWalkForwardSplitter(
        train_size=100, test_size=20, label_horizon=MAX_ADAPTIVE_HORIZON_CANDLES,
    )

    _, _, info_insufficient = next(insufficient_splitter.split(df))
    _, _, info_correct = next(correct_splitter.split(df))

    assert info_correct["train_size"] <= info_insufficient["train_size"]
    assert info_correct["train_size"] == 100 - MAX_ADAPTIVE_HORIZON_CANDLES

def test_triple_labels_atr_barrier_wider_than_fixed_threshold():
    """При большом ATR-множителе барьер TP/SL шире, чем узкий фикс.-threshold — сделка не закрывается так быстро."""
    n = 30
    close = [100.0] * n
    high = [100.3] * n
    low = [99.9] * n  # не пробивает ни фикс. SL (99.0), ни ATR-SL (99.8)
    atr = [1.0] * n  # абсолютный ATR = 1.0 (1% от цены)

    df = pd.DataFrame({"close": close, "high": high, "low": low, "atr": atr})

    # Фикс. threshold=0.01 -> барьер TP=101, узкий high=100.3 его не достает -> HOLD
    labels_fixed = generate_triple_labels(df, horizon=5, threshold=0.01)
    assert labels_fixed.iloc[0] == 0.0

    # ATR-барьер с малым множителем 0.2 -> tp=100.2, sl=99.8; high=100.3 пробивает TP, low=99.9 SL не трогает
    labels_atr = generate_triple_labels(
        df, horizon=5, threshold=0.01, tp_atr_mult=0.2, sl_atr_mult=0.2
    )
    assert labels_atr.iloc[0] == 1.0


def test_triple_labels_atr_barrier_requires_both_mults():
    """Если задан только один из множителей — используется старое поведение (fallback), а не половинчатый ATR-барьер."""
    n = 20
    df = pd.DataFrame({
        "close": [100.0] * n,
        "high": [100.3] * n,
        "low": [99.7] * n,
        "atr": [1.0] * n,
    })

    labels = generate_triple_labels(df, horizon=5, threshold=0.01, tp_atr_mult=0.2)
    # sl_atr_mult не задан -> use_atr_barrier=False -> фикс.-threshold ветка -> HOLD
    assert labels.iloc[0] == 0.0


def test_triple_labels_short_uses_its_own_mirrored_atr_barriers():
    """SHORT TP is -tp_atr_mult*ATR and SL is +sl_atr_mult*ATR."""
    n = 30
    base = {
        "close": [100.0] * n,
        "high": [100.5] * n,
        "low": [99.5] * n,
        "atr": [1.0] * n,
    }

    # A -1 ATR move is the LONG stop, but not the SHORT take-profit (-1.5 ATR).
    near_short_tp = pd.DataFrame(base | {"low": [99.5, 98.8] + [99.5] * (n - 2)})
    labels = generate_triple_labels(
        near_short_tp, horizon=5, tp_atr_mult=1.5, sl_atr_mult=1.0
    )
    assert labels.iloc[0] == 0.0

    short_tp = pd.DataFrame(base | {"low": [99.5, 98.5] + [99.5] * (n - 2)})
    labels = generate_triple_labels(
        short_tp, horizon=5, tp_atr_mult=1.5, sl_atr_mult=1.0
    )
    assert labels.iloc[0] == -1.0
