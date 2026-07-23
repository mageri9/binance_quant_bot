import pytest
import pandas as pd
import numpy as np

from src.features.engineering import OHLCV_FEATURES, add_features


def test_add_features_success():
    # Создаем тестовую таблицу с ценами на 50 свечей для надежного разогрева скользящих окон
    np.random.seed(42)
    # Имитируем плавный рост цены с 100 до 150
    close_prices = np.linspace(100, 150, 50)
    volumes = np.random.uniform(1000, 2000, 50)

    dummy_data = {
        "open_time": np.arange(1000, 1050),
        "open": close_prices - 1.0,
        "high": close_prices + 2.0,
        "low": close_prices - 2.0,
        "close": close_prices,
        "volume": volumes,
    }

    df = pd.DataFrame(dummy_data)

    # Вызываем расчет признаков
    df_features = add_features(df)

    # Проверяем, появились ли все старые и новые колонки
    expected_cols = [
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
        "bb_upper_pct",
        "bb_middle_pct",
        "bb_lower_pct",
        "atr_pct",
        "macd_pct",
        "macd_signal_pct",
        "macd_hist_pct",
    ]
    for col in expected_cols:
        assert col in df_features.columns

    for col in OHLCV_FEATURES:
        assert col in df_features.columns

    # Из-за оконных функций первые строки будут содержать NaN.
    # Проверяем, что к 30-й строке все новые значения успешно рассчитались.
    row_30 = df_features.iloc[30]
    assert not np.isnan(row_30["rsi"])
    assert not np.isnan(row_30["macd"])
    assert not np.isnan(row_30["bb_upper"])
    assert not np.isnan(row_30["atr"])
    assert not np.isnan(row_30["adx"])
    assert not np.isnan(row_30["atr_pct"])
    assert not np.isnan(row_30["macd_pct"])
    assert not np.isnan(row_30["macd_signal_pct"])
    assert not np.isnan(row_30["macd_hist_pct"])
    assert not np.isnan(row_30["bb_upper_pct"])
    assert not np.isnan(row_30["bb_middle_pct"])
    assert not np.isnan(row_30["bb_lower_pct"])

    # Математические проверки на ограничения и логику новых признаков:
    # 1. Линии Боллинджера: верхняя полоса должна быть строго выше средней, а средняя — выше нижней.
    assert (df_features["bb_upper"].dropna() >= df_features["bb_middle"].dropna()).all()
    assert (df_features["bb_middle"].dropna() >= df_features["bb_lower"].dropna()).all()

    # 2. Волатильность и ATR не могут принимать отрицательные значения
    assert (df_features["volatility"].dropna() >= 0).all()
    assert (df_features["atr"].dropna() >= 0).all()

    # 3. Индекс ADX математически ограничен пределами от 0 до 100
    assert df_features["adx"].dropna().between(0, 100).all()

    # 4. Нормализованный ATR (в процентах от цены) не может быть отрицательным
    assert (df_features["atr_pct"].dropna() >= 0).all()

    # 5. Нормализованные признаки должны быть стационарны — разумный диапазон
    #    относительно цены (не должны принимать абсурдные значения на плавном тренде)
    assert df_features["bb_upper_pct"].dropna().abs().max() < 1.0
    assert df_features["bb_lower_pct"].dropna().abs().max() < 1.0
    assert df_features["bb_middle_pct"].dropna().abs().max() < 1.0
    assert df_features["atr_pct"].dropna().abs().max() < 1.0
    assert df_features["macd_pct"].dropna().abs().max() < 1.0

    assert df_features["close_location"].dropna().between(0, 1).all()
    assert (df_features["hl_range_pct"].dropna() >= 0).all()
    assert (df_features[["rv_10", "rv_20"]].dropna() >= 0).all().all()


def test_ohlcv_features_do_not_use_future_candles():
    close = np.linspace(100, 220, 80)
    df = pd.DataFrame({
        "open": close - 1, "high": close + 2, "low": close - 2,
        "close": close, "volume": np.linspace(1000, 2000, 80),
    })
    original = add_features(df)
    changed = df.copy()
    changed.loc[60:, ["open", "high", "low", "close", "volume"]] *= 5
    recomputed = add_features(changed)

    pd.testing.assert_frame_equal(
        original.loc[:59, OHLCV_FEATURES], recomputed.loc[:59, OHLCV_FEATURES],
    )


def test_drift_detection_success():
    from src.features.drift import ConceptDriftDetector
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    # Создаем два датасета с разным распределением признака 'rsi'
    ref_df = pd.DataFrame({"rsi": np.random.normal(loc=50, scale=10, size=100)})
    cur_df = pd.DataFrame({"rsi": np.random.normal(loc=70, scale=10, size=100)})

    report = ConceptDriftDetector.detect_drift(ref_df, cur_df, ["rsi"])

    assert report["drift_detected"] == True
    assert "rsi" in report["metrics"]
    assert report["metrics"]["rsi"]["drift"] == True
