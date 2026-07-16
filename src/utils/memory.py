import numpy as np
import pandas as pd


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит числовые колонки DataFrame к наименее затратным по памяти типам:
      - float64 -> float32 (индикаторы, цены, метки не требуют double-точности)
      - int64/int32 -> наименьший подходящий целочисленный тип
        (pd.to_numeric сам выбирает по фактическому диапазону значений,
        поэтому большие open_time в миллисекундах не обрезаются)

    Изменяет df на месте и возвращает его же для удобства чейнинга.
    """
    if df.empty:
        return df

    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

    int_cols = df.select_dtypes(include=["int64", "int32"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df