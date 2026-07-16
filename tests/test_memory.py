import numpy as np
import pandas as pd

from src.utils.memory import downcast_dtypes


def test_downcast_dtypes_reduces_float_memory():
    df = pd.DataFrame({
        "a": np.array([1.5, 2.5, 3.5], dtype=np.float64),
        "b": np.array([1, 2, 3], dtype=np.int64),
    })
    mem_before = df.memory_usage(deep=True).sum()

    result = downcast_dtypes(df)

    assert result["a"].dtype == np.float32
    assert result["b"].dtype in (np.int8, np.int16, np.int32)
    assert result.memory_usage(deep=True).sum() < mem_before


def test_downcast_dtypes_preserves_large_int64_values():
    # open_time в миллисекундах — далеко за пределами int32, не должен обрезаться
    big_ts = 1_700_000_000_000
    df = pd.DataFrame({"open_time": [big_ts, big_ts + 3600000]})

    result = downcast_dtypes(df)

    assert result["open_time"].dtype == np.int64
    assert result["open_time"].iloc[0] == big_ts


def test_downcast_dtypes_handles_nan_float_column():
    df = pd.DataFrame({"target": [1.0, np.nan, -1.0]})
    result = downcast_dtypes(df)
    assert result["target"].dtype == np.float32
    assert np.isnan(result["target"].iloc[1])


def test_downcast_dtypes_empty_dataframe():
    df = pd.DataFrame()
    result = downcast_dtypes(df)
    assert result.empty


def test_add_features_output_is_downcast():
    from src.features.engineering import add_features

    n = 40
    close = np.linspace(100, 110, n)
    df = pd.DataFrame({
        "open_time": np.arange(n),
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.uniform(1000, 2000, n),
    })

    df_feats = add_features(df)

    for col in ["rsi", "macd", "atr", "adx"]:
        assert df_feats[col].dtype == np.float32