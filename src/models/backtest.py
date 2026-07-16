import pandas as pd
from typing import Generator, Tuple, Dict, Any


def purge_train_tail(train_df: pd.DataFrame, purge_rows: int) -> pd.DataFrame:
    """
    Отбрасывает последние `purge_rows` строк train_df.

    Используется двумя способами:
      1) внутри TimeSeriesWalkForwardSplitter — покадрово, перед каждым
         test-окном фолда;
      2) отдельно в run_lgbm_experiment — на границе df_train_val/df_holdout,
         где train/test режутся вручную, а не через сплиттер.

    Оба случая защищают от одной и той же утечки: адаптивный горизонт
    Triple Barrier может резолвить метку данными, лежащими уже за
    границей train-окна.
    """
    if purge_rows <= 0:
        return train_df
    if len(train_df) <= purge_rows:
        return train_df.iloc[0:0]
    return train_df.iloc[:-purge_rows]


class TimeSeriesWalkForwardSplitter:
    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        label_horizon: int = 0,
    ):
        """
        :param label_horizon: purge — количество последних строк train_df,
            которые отбрасываются перед обучением. ВАЖНО: это значение
            должно отражать МАКСИМАЛЬНО ВОЗМОЖНЫЙ горизонт разметки, а не
            номинальный (настроенный) horizon — при адаптивной Triple
            Barrier разметке (см. src.labels.generator.MAX_ADAPTIVE_HORIZON_CANDLES)
            фактический горизонт метки может быть значительно больше
            конфигурационного LABEL_HORIZON. Недостаточный purge означает
            утечку данных из test в train. При 0 (по умолчанию) поведение
            не меняется — обратная совместимость для случаев без такой
            разметки.
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

            train_df = purge_train_tail(train_df, self.label_horizon)

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