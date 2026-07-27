import numpy as np
import pandas as pd


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def calculate_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd_line = fast - slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_prev = df["high"].shift(1)
    low_prev = df["low"].shift(1)
    tr = calculate_atr(df, period)

    up_move = df["high"] - high_prev
    down_move = low_prev - df["low"]

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index).ewm(com=period-1, adjust=False).mean()
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index).ewm(com=period-1, adjust=False).mean()

    plus_di = 100 * (plus_dm / (tr + 1e-8))
    minus_di = 100 * (minus_dm / (tr + 1e-8))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    return dx.ewm(com=period - 1, adjust=False).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генератор признаков для модели машинного обучения."""
    df_out = df.copy()
    close = df_out["close"]

    df_out["rsi"] = calculate_rsi(close)
    macd, macd_sig, macd_hist = calculate_macd(close)
    df_out["macd_pct"] = macd / close
    df_out["macd_signal_pct"] = macd_sig / close
    df_out["macd_hist_pct"] = macd_hist / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df_out["bb_middle_pct"] = (close - bb_mid) / close
    df_out["bb_upper_pct"] = (bb_mid + 2 * bb_std - close) / close
    df_out["bb_lower_pct"] = (close - (bb_mid - 2 * bb_std)) / close

    atr = calculate_atr(df_out)
    df_out["atr"] = atr
    df_out["atr_pct"] = atr / close
    df_out["adx"] = calculate_adx(df_out)

    df_out["volatility"] = close.pct_change().rolling(14).std()
    df_out["volume_ratio"] = df_out["volume"] / (df_out["volume"].rolling(14).mean() + 1e-8)
    df_out["return_1"] = close.pct_change(1)
    df_out["return_3"] = close.pct_change(3)

    float_columns = df_out.select_dtypes(include=[np.floating]).columns
    df_out[float_columns] = df_out[float_columns].astype(np.float32)
    return df_out
