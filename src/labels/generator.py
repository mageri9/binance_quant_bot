import pandas as pd
import numpy as np


def generate_binary_labels(
    df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0
) -> pd.Series:
    """
    Генерирует бинарную метку направления цены через 'horizon' свечей.

    1 - цена закроется выше текущей более чем на threshold (выраженный долей, например 0.01 для 1%)
    0 - цена закроется ниже или изменение будет меньше threshold

    Последние 'horizon' строк будут иметь значение NaN, так как будущее для них еще не наступило.
    """
    # Сдвигаем цены закрытия назад (заглядываем в будущее на horizon шагов)
    future_close = df["close"].shift(-horizon)

    # Считаем доходность относительно текущей цены закрытия
    future_return = (future_close - df["close"]) / df["close"]

    # Создаем бинарную метку
    label = (future_return > threshold).astype(float)

    # Для последних 'horizon' свечей будущее неизвестно — ставим NaN
    label.iloc[-horizon:] = np.nan

    return label


def generate_triple_labels(
    df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01
) -> pd.Series:
    """
    Генерирует тройную метку для торговли:
    1.0 (Long)   - будущая цена вырастет более чем на threshold
    -1.0 (Short) - будущая цена упадет более чем на threshold
    0.0 (Hold)   - цена останется в пределах коридора [-threshold, threshold]

    Последние 'horizon' строк будут иметь значение NaN.
    """
    future_close = df["close"].shift(-horizon)
    future_return = (future_close - df["close"]) / df["close"]

    # Инициализируем метку нулями (Hold)
    label = pd.Series(0.0, index=df.index, dtype=float)

    # Заполняем Long и Short сигналы
    label.loc[future_return > threshold] = 1.0
    label.loc[future_return < -threshold] = -1.0

    # Последние строки заполняем NaN
    label.iloc[-horizon:] = np.nan

    return label