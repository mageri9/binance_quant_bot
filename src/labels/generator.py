import pandas as pd
import numpy as np


MIN_ADAPTIVE_HORIZON_CANDLES = 2
MAX_ADAPTIVE_HORIZON_CANDLES = 15


def generate_binary_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
    tp_atr_mult: float | None = None,
) -> pd.Series:
    """
    Генерирует бинарную метку направления цены.
    С поддержкой фолбэка для простых тестов.

    tp_atr_mult: если задан, барьер TP считается как close + tp_atr_mult * ATR
    вместо фиксированного close * (1 + threshold). Требует наличия колонки 'atr'.
    """
    if "high" not in df.columns or "low" not in df.columns or "atr" not in df.columns:
        # Упрощенный фолбэк для совместимости со старыми тестами
        future_close = df["close"].shift(-horizon)
        future_return = (future_close - df["close"]) / df["close"]
        label = (future_return > threshold).astype(float)
        label.iloc[-horizon:] = np.nan
        return label

    # Адаптивная бинарная разметка на основе волатильности
    n = len(df)
    labels = np.zeros(n, dtype=float)
    close = df["close"].values
    high = df["high"].values
    atr = df["atr"].values

    atr_series = pd.Series(atr)
    atr_rolling = atr_series.rolling(window=100, min_periods=1).mean().values

    safe_end = n - MAX_ADAPTIVE_HORIZON_CANDLES

    for t in range(safe_end):
        curr_atr = atr[t]
        curr_atr_rolling = atr_rolling[t]

        if np.isnan(curr_atr) or np.isnan(curr_atr_rolling) or curr_atr == 0:
            hz = horizon
        else:
            ratio = curr_atr_rolling / curr_atr
            hz = int(
                np.clip(
                    np.round(horizon * ratio),
                    MIN_ADAPTIVE_HORIZON_CANDLES,
                    MAX_ADAPTIVE_HORIZON_CANDLES,
                )
            )

        if t + hz >= n:
            labels[t] = np.nan
            continue

        p_close = close[t]

        if tp_atr_mult is not None and not np.isnan(curr_atr) and curr_atr > 0:
            tp_barrier = p_close + tp_atr_mult * curr_atr
        else:
            tp_barrier = p_close * (1.0 + threshold)

        hit_tp = False
        for k in range(t + 1, t + hz + 1):
            if high[k] >= tp_barrier:
                hit_tp = True
                break

        labels[t] = 1.0 if hit_tp else 0.0

    labels[safe_end:] = np.nan

    return pd.Series(labels, index=df.index)


def generate_triple_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.01,
    tp_atr_mult: float | None = None,
    sl_atr_mult: float | None = None,
) -> pd.Series:
    """
    Генерирует профессиональную трехклассовую разметку методом Triple Barrier
    с адаптивным временным горизонтом на основе волатильности ATR.

    tp_atr_mult / sl_atr_mult: если оба заданы, барьеры TP/SL считаются как
    close +/- mult * ATR вместо фиксированного close * (1 +/- threshold).
    Требует наличия колонки 'atr'. Заданы независимо — но для консистентной
    экономики стратегии их следует калибровать вместе (см. scripts/calibrate.py).
    """
    if "high" not in df.columns or "low" not in df.columns or "atr" not in df.columns:
        # Упрощенный фолбэк для совместимости со старыми тестами
        future_close = df["close"].shift(-horizon)
        future_return = (future_close - df["close"]) / df["close"]
        label = pd.Series(0.0, index=df.index, dtype=float)
        label.loc[future_return > threshold] = 1.0
        label.loc[future_return < -threshold] = -1.0
        label.iloc[-horizon:] = np.nan
        return label

    n = len(df)
    labels = np.zeros(n, dtype=float)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values

    atr_series = pd.Series(atr)
    atr_rolling = atr_series.rolling(window=100, min_periods=1).mean().values

    safe_end = n - MAX_ADAPTIVE_HORIZON_CANDLES
    use_atr_barrier = tp_atr_mult is not None and sl_atr_mult is not None

    for t in range(safe_end):
        curr_atr = atr[t]
        curr_atr_rolling = atr_rolling[t]

        if np.isnan(curr_atr) or np.isnan(curr_atr_rolling) or curr_atr == 0:
            hz = horizon
        else:
            ratio = curr_atr_rolling / curr_atr
            hz = int(
                np.clip(
                    np.round(horizon * ratio),
                    MIN_ADAPTIVE_HORIZON_CANDLES,
                    MAX_ADAPTIVE_HORIZON_CANDLES,
                )
            )

        # Дополнительная защита на случай изменения констант
        if t + hz >= n:
            labels[t] = np.nan
            continue

        p_close = close[t]

        if use_atr_barrier and not np.isnan(curr_atr) and curr_atr > 0:
            tp_barrier = p_close + tp_atr_mult * curr_atr
            sl_barrier = p_close - sl_atr_mult * curr_atr
        else:
            tp_barrier = p_close * (1.0 + threshold)
            sl_barrier = p_close * (1.0 - threshold)

        tp_idx = -1
        sl_idx = -1

        for k in range(t + 1, t + hz + 1):
            if tp_idx == -1 and high[k] >= tp_barrier:
                tp_idx = k
            if sl_idx == -1 and low[k] <= sl_barrier:
                sl_idx = k

        if tp_idx != -1 and (sl_idx == -1 or tp_idx < sl_idx):
            labels[t] = 1.0
        elif sl_idx != -1 and (tp_idx == -1 or sl_idx < tp_idx):
            labels[t] = -1.0
        else:
            labels[t] = 0.0

    # Хвост всегда остается неразмеченным
    labels[safe_end:] = np.nan

    return pd.Series(labels, index=df.index)