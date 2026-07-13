import pytest
import pandas as pd
import numpy as np

from src.features.engineering import add_features


def test_add_features_success():
    # Создаем тестовую таблицу с ценами на 30 свечей
    np.random.seed(42)
    # Имитируем плавный рост цены с 100 до 130
    close_prices = np.linspace(100, 130, 30)
    volumes = np.random.uniform(1000, 2000, 30)

    dummy_data = {
        "open_time": np.arange(1000, 1030),
        "open": close_prices - 1.0,
        "high": close_prices + 2.0,
        "low": close_prices - 2.0,
        "close": close_prices,
        "volume": volumes,
    }

    df = pd.DataFrame(dummy_data)

    # Вызываем расчет признаков
    df_features = add_features(df)

    # Проверяем, появились ли колонки
    expected_cols = [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility",
        "volume_ratio",
    ]
    for col in expected_cols:
        assert col in df_features.columns

    # Из-за оконных функций (rolling, ewm) первые строки будут содержать NaN
    # Проверяем, что к 20-й строке все значения рассчитаны
    row_20 = df_features.iloc[20]
    assert not np.isnan(row_20["rsi"])
    assert not np.isnan(row_20["macd"])
    assert not np.isnan(row_20["volatility"])
    assert not np.isnan(row_20["volume_ratio"])

    # Проверяем математические ограничения индикаторов
    # RSI всегда строго в пределах [0, 100]
    assert df_features["rsi"].dropna().between(0, 100).all()

    # Волатильность не может быть отрицательной
    assert (df_features["volatility"].dropna() >= 0).all()

    # Так как цена постоянно росла, RSI на последних свечах должен быть высоким (> 50)
    assert df_features["rsi"].iloc[-1] > 50