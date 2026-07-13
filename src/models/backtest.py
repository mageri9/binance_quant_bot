import pandas as pd
from typing import Generator, Tuple, Dict, Any


class TimeSeriesWalkForwardSplitter:
    """
    Класс для честного разделения временных рядов (Walk-Forward Validation).
    Обучение происходит строго на прошлом, тестирование — строго на будущем.
    """

    def __init__(self, train_size: int, test_size: int, step_size: int | None = None):
        """
        :param train_size: Количество свечей для обучения модели.
        :param test_size: Количество свечей для тестирования модели.
        :param step_size: На сколько свечей сдвигается окно на каждом шаге.
                          Если равен None, то равен test_size (тесты идут стык в стык).
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size

    def split(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Разрезает DataFrame на наборы (фолды) для обучения и теста.
        Возвращает генератор кортежей: (таблица_обучения, таблица_теста, информация_о_шаге)
        """
        n_samples = len(df)

        # Если данных слишком мало даже для одного шага — ничего не возвращаем
        if n_samples < (self.train_size + self.test_size):
            return

        start_idx = 0
        fold = 0

        while True:
            train_start = start_idx
            train_end = start_idx + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            # Если тестовое окно выходит за рамки имеющихся данных — останавливаемся
            if test_end > n_samples:
                break

            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            info = {
                "fold": fold,
                "train_start_idx": train_start,
                "train_end_idx": train_end,
                "test_start_idx": test_start,
                "test_end_idx": test_end,
                "train_size": len(train_df),
                "test_size": len(test_df),
            }

            yield train_df, test_df, info

            # Сдвигаем окно вперед на заданный шаг
            start_idx += self.step_size
            fold += 1