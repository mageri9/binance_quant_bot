import pandas as pd
import numpy as np

from src.utils.memory import downcast_dtypes


# Families are kept explicit so offline ablations can compare them against the
# established technical-indicator baseline without changing their definitions.
TECHNICAL_FEATURES = [
    "rsi", "macd_pct", "macd_signal_pct", "macd_hist_pct", "volatility",
    "volume_ratio", "bb_upper_pct", "bb_middle_pct", "bb_lower_pct", "atr_pct", "adx",
]
OHLCV_FEATURE_GROUPS = {
    "lagged_returns": ["return_1", "return_3", "return_6"],
    "range_location": ["hl_range_pct", "close_location"],
    "realized_volatility": ["rv_10", "rv_20"],
    "volume_surprise": ["volume_surprise_20"],
    "trend_slopes": ["trend_slope_10", "trend_slope_20"],
}
OHLCV_FEATURES = [feature for group in OHLCV_FEATURE_GROUPS.values() for feature in group]
DEFAULT_FEATURE_SCHEMA = TECHNICAL_FEATURES + OHLCV_FEATURES


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Рассчитывает индекс относительной силы (RSI) сглаживанием по методу Уайлдера.
    Возвращает значения от 0 до 100.
    """
    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Рассчитывает линии индикатора MACD, Сигнальную линию и Гистограмму.
    """
    fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return macd_line, signal_line, macd_hist


def calculate_volatility(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Рассчитывает историческую волатильность цены за период.
    """
    returns = df["close"].pct_change()
    volatility = returns.rolling(window=period).std()
    return volatility


def calculate_volume_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Показывает отношение текущего объема к скользящему среднему объему.
    """
    avg_volume = df["volume"].rolling(window=period).mean()
    ratio = df["volume"] / (avg_volume + 1e-8)
    return ratio


def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Рассчитывает Линии Боллинджера (Bollinger Bands).
    Возвращает (BB_Upper, BB_Middle, BB_Lower).
    """
    middle_band = df["close"].rolling(window=period).mean()
    std_dev = df["close"].rolling(window=period).std()
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)
    return upper_band, middle_band, lower_band


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Рассчитывает средний истинный диапазон (ATR, Average True Range).
    Использует сглаживание Уайлдера.
    """
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()

    # Истинный диапазон (True Range, TR)
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # ATR - сглаженное среднее от TR
    atr = tr.ewm(com=period - 1, adjust=False).mean()
    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Рассчитывает индекс среднего направленного движения (ADX, Average Directional Index).
    """
    high_prev = df["high"].shift(1)
    low_prev = df["low"].shift(1)
    close_prev = df["close"].shift(1)

    # Вычисление True Range (TR)
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - close_prev).abs()
    low_close_prev = (df["low"] - close_prev).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # Сглаженный TR
    tr_smoothed = tr.ewm(com=period - 1, adjust=False).mean()

    # Вычисление Directional Movement (+DM, -DM)
    up_move = df["high"] - high_prev
    down_move = low_prev - df["low"]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Сглаженные DM
    plus_dm_smoothed = (
        pd.Series(plus_dm, index=df.index).ewm(com=period - 1, adjust=False).mean()
    )
    minus_dm_smoothed = (
        pd.Series(minus_dm, index=df.index).ewm(com=period - 1, adjust=False).mean()
    )

    # Directional Indicators (+DI, -DI)
    plus_di = 100 * (plus_dm_smoothed / (tr_smoothed + 1e-8))
    minus_di = 100 * (minus_dm_smoothed / (tr_smoothed + 1e-8))

    # Directional Index (DX)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)

    # ADX - сглаженное среднее от DX
    adx = dx.ewm(com=period - 1, adjust=False).mean()
    return adx


def calculate_trend_slope(close: pd.Series, period: int) -> pd.Series:
    """Return the rolling least-squares slope of log-price per candle."""
    log_close = np.log(close.where(close > 0))
    x = np.arange(period, dtype=float)
    sum_x = x.sum()
    denominator = period * np.square(x).sum() - sum_x ** 2
    sum_y = log_close.rolling(period).sum()
    sum_xy = log_close.rolling(period).apply(lambda values: np.dot(x, values), raw=True)
    return (period * sum_xy - sum_x * sum_y) / denominator


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Принимает DataFrame со свечами (open, high, low, close, volume)
    и возвращает новый DataFrame со всеми рассчитанными признаками.
    """
    df_out = df.copy()

    if "open_time" in df_out.columns:
        df_out = df_out.sort_values("open_time").reset_index(drop=True)

    # 1. Рассчитываем базовые признаки
    df_out["rsi"] = calculate_rsi(df_out)

    macd_line, signal_line, macd_hist = calculate_macd(df_out)
    df_out["macd"] = macd_line
    df_out["macd_signal"] = signal_line
    df_out["macd_hist"] = macd_hist

    df_out["volatility"] = calculate_volatility(df_out)
    df_out["volume_ratio"] = calculate_volume_ratio(df_out)

    # 2. Рассчитываем новые признаки (Bollinger Bands, ATR, ADX)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df_out)
    df_out["bb_upper"] = bb_upper
    df_out["bb_middle"] = bb_middle
    df_out["bb_lower"] = bb_lower

    df_out["atr"] = calculate_atr(df_out)
    df_out["adx"] = calculate_adx(df_out)

    # 3. Нормализованные (стационарные) версии ценовых индикаторов — для модели.
    # Абсолютные версии выше оставлены как есть: atr используется в src/labels/generator.py
    # для расчета ценовых барьеров, остальные — для обратной совместимости.
    df_out["bb_upper_pct"] = (bb_upper - df_out["close"]) / df_out["close"]
    df_out["bb_middle_pct"] = (df_out["close"] - bb_middle) / df_out["close"]
    df_out["bb_lower_pct"] = (df_out["close"] - bb_lower) / df_out["close"]

    df_out["atr_pct"] = df_out["atr"] / df_out["close"]

    df_out["macd_pct"] = macd_line / df_out["close"]
    df_out["macd_signal_pct"] = signal_line / df_out["close"]
    df_out["macd_hist_pct"] = macd_hist / df_out["close"]

    # OHLCV-only candidates for the P2 ablation.  Every value at t uses the
    # completed candle t and earlier observations, never a future candle.
    returns = df_out["close"].pct_change()
    df_out["return_1"] = returns
    df_out["return_3"] = df_out["close"].pct_change(3)
    df_out["return_6"] = df_out["close"].pct_change(6)
    df_out["hl_range_pct"] = (df_out["high"] - df_out["low"]) / df_out["close"]
    candle_range = df_out["high"] - df_out["low"]
    df_out["close_location"] = np.where(
        candle_range > 0,
        (df_out["close"] - df_out["low"]) / candle_range,
        0.5,
    )
    df_out["rv_10"] = np.sqrt(returns.pow(2).rolling(10).sum())
    df_out["rv_20"] = np.sqrt(returns.pow(2).rolling(20).sum())

    log_volume = np.log1p(df_out["volume"].clip(lower=0))
    volume_mean = log_volume.rolling(20).mean().shift(1)
    volume_std = log_volume.rolling(20).std().shift(1)
    df_out["volume_surprise_20"] = (log_volume - volume_mean) / (volume_std + 1e-8)
    df_out["trend_slope_10"] = calculate_trend_slope(df_out["close"], 10)
    df_out["trend_slope_20"] = calculate_trend_slope(df_out["close"], 20)

    return downcast_dtypes(df_out)
