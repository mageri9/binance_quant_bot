import pytest
import pandas as pd
import numpy as np

from src.features.engineering import add_features


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
    ]
    for col in expected_cols:
        assert col in df_features.columns

    # Из-за оконных функций первые строки будут содержать NaN.
    # Проверяем, что к 30-й строке все новые значения успешно рассчитались.
    row_30 = df_features.iloc[30]
    assert not np.isnan(row_30["rsi"])
    assert not np.isnan(row_30["macd"])
    assert not np.isnan(row_30["bb_upper"])
    assert not np.isnan(row_30["atr"])
    assert not np.isnan(row_30["adx"])

    # Математические проверки на ограничения и логику новых признаков:
    # 1. Линии Боллинджера: верхняя полоса должна быть строго выше средней, а средняя — выше нижней.
    assert (df_features["bb_upper"].dropna() >= df_features["bb_middle"].dropna()).all()
    assert (df_features["bb_middle"].dropna() >= df_features["bb_lower"].dropna()).all()

    # 2. Волатильность и ATR не могут принимать отрицательные значения
    assert (df_features["volatility"].dropna() >= 0).all()
    assert (df_features["atr"].dropna() >= 0).all()

    # 3. Индекс ADX математически ограничен пределами от 0 до 100
    assert df_features["adx"].dropna().between(0, 100).all()