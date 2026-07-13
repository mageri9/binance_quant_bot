import pandas as pd
import numpy as np


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Рассчитывает индекс относительной силы (RSI) сглаживанием по методу Уайлдера.
    Возвращает значения от 0 до 100.
    """
    # Вычисляем разницу цен закрытия между соседними свечами
    delta = df["close"].diff()

    # Разделяем движения вверх (gain) и вниз (loss)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Применяем экспоненциальное сглаживание Уайлдера
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    # Предотвращаем деление на ноль
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
    Рассчитывает историческую волатильность цены на основе процентной доходности за период.
    """
    returns = df["close"].pct_change()
    volatility = returns.rolling(window=period).std()
    return volatility


def calculate_volume_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Показывает отношение текущего объема к скользящему среднему объему.
    Значение > 1 означает всплеск активности.
    """
    avg_volume = df["volume"].rolling(window=period).mean()
    ratio = df["volume"] / (avg_volume + 1e-8)
    return ratio


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Принимает DataFrame со свечами (open, high, low, close, volume)
    и возвращает новый DataFrame со всеми рассчитанными признаками.
    """
    # Работаем с копией, чтобы не менять исходную таблицу (чистая функция)
    df_out = df.copy()

    # Убеждаемся, что данные отсортированы по времени
    if "open_time" in df_out.columns:
        df_out = df_out.sort_values("open_time").reset_index(drop=True)

    df_out["rsi"] = calculate_rsi(df_out)

    macd_line, signal_line, macd_hist = calculate_macd(df_out)
    df_out["macd"] = macd_line
    df_out["macd_signal"] = signal_line
    df_out["macd_hist"] = macd_hist

    df_out["volatility"] = calculate_volatility(df_out)
    df_out['volume_ratio'] = calculate_volume_ratio(df_out)

    return df_out