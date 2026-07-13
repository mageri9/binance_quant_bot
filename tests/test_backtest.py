import pytest
import pandas as pd
import numpy as np

from src.models.backtest import TimeSeriesWalkForwardSplitter


def test_walk_forward_splitter_success():
    # Создаем искусственную таблицу на 100 строк
    df = pd.DataFrame(
        {"timestamp": np.arange(100), "value": np.random.uniform(10, 20, 100)}
    )

    # Задаем параметры:
    # 50 строк на обучение, 10 строк на тест, шаг сдвига 10 строк.
    splitter = TimeSeriesWalkForwardSplitter(train_size=50, test_size=10, step_size=10)

    splits = list(splitter.split(df))

    # 100 строк всего. Минус 50 на обучение = остается 50.
    # Так как размер теста 10 и шаг 10, мы должны получить ровно 5 фолдов (наборов)
    assert len(splits) == 5

    for i, (train_df, test_df, info) in enumerate(splits):
        assert info["fold"] == i
        assert len(train_df) == 50
        assert len(test_df) == 10

        # Проверяем строгий хронологический порядок:
        # Самое последнее время в обучении строго меньше самого первого времени в тесте
        assert train_df["timestamp"].max() < test_df["timestamp"].min()

        # Проверяем, что индексы в метаданных рассчитаны абсолютно верно
        assert info["train_start_idx"] == i * 10
        assert info["train_end_idx"] == i * 10 + 50
        assert info["test_start_idx"] == i * 10 + 50
        assert info["test_end_idx"] == i * 10 + 60


def test_walk_forward_splitter_insufficient_data():
    df = pd.DataFrame({"value": [1, 2, 3]})
    # Если данных слишком мало, нарезка производиться не должна
    splitter = TimeSeriesWalkForwardSplitter(train_size=10, test_size=5)
    splits = list(splitter.split(df))
    assert len(splits) == 0