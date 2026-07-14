import pandas as pd
from typing import Generator, Tuple, Dict, Any


class TimeSeriesWalkForwardSplitter:
    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        label_horizon: int = 0,
    ):
        """
        :param label_horizon: количество последних строк train_df,
            которые отбрасываются перед обучением. Нужно, если метки
            (target) размечены через shift(-horizon) на полном
            датафрейме ДО разбиения на train/test — тогда последние
            `label_horizon` строк train содержат метку, посчитанную
            по цене закрытия, физически лежащей уже в test-окне
            следующего сегмента (утечка будущего). При 0 (по
            умолчанию) поведение не меняется — обратная совместимость
            для случаев, где сплиттер используется без такой разметки.
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.label_horizon = label_horizon

    def split(self, df: pd.DataFrame):
        n_samples = len(df)
        if n_samples < (self.train_size + self.test_size):
            return

        start_idx = 0
        fold = 0

        while True:
            train_start = start_idx
            train_end = start_idx + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            # Отбрасываем последние label_horizon строк train_df: их метка
            # рассчитана по close, лежащему уже в тестовом окне (см. docstring).
            if self.label_horizon > 0:
                train_df = train_df.iloc[: -self.label_horizon] if len(train_df) > self.label_horizon else train_df.iloc[0:0]

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

            start_idx += self.step_size
            fold += 1