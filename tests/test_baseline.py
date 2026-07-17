import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import select

from src.db.models import Experiment
from src.models.baseline import run_baseline_experiment


@pytest.mark.asyncio
async def test_run_baseline_experiment_success(temp_db_session):
    # 1. Генерируем тестовый датасет на 150 строк
    np.random.seed(42)
    n_rows = 150

    dummy_data = {
        "open_time": np.arange(1000, 1000 + n_rows),
        "open": np.random.uniform(100, 110, n_rows),
        "high": np.random.uniform(110, 120, n_rows),
        "low": np.random.uniform(90, 100, n_rows),
        "close": np.random.uniform(100, 110, n_rows),
        "volume": np.random.uniform(1000, 5000, n_rows),
        # Рассчитанные признаки
        "rsi": np.random.uniform(20, 80, n_rows),
        "macd": np.random.uniform(-1, 1, n_rows),
        "macd_signal": np.random.uniform(-1, 1, n_rows),
        "macd_hist": np.random.uniform(-1, 1, n_rows),
        "volatility": np.random.uniform(0.01, 0.05, n_rows),
        "volume_ratio": np.random.uniform(0.5, 2.0, n_rows),
        # Метка направления цены
        "target_binary": np.random.choice([0.0, 1.0], size=n_rows),
    }
    df = pd.DataFrame(dummy_data)

    metadata = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "version": "1.0-test",
        "features": [
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "volatility",
            "volume_ratio",
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "test_dataset.parquet")
        metadata_path = os.path.join(tmpdir, "test_metadata.json")

        df.to_parquet(dataset_path, index=False)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        # 2. Запускаем эксперимент с уменьшенными размерами окон (чтобы уложиться в 150 строк)
        result = await run_baseline_experiment(
            session=temp_db_session,
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            train_size=100,
            test_size=20,
            c_parameter=1.0,
        )

        # 3. Проверяем возвращаемый результат
        assert "experiment_id" in result
        assert result["parameters"]["model_type"] == "LogisticRegression"
        assert "accuracy" in result["metrics"]
        assert result["metrics"]["total_folds"] > 0

        # 4. Проверяем, что в базе данных действительно появилась запись об эксперименте
        stmt = select(Experiment).where(Experiment.id == result["experiment_id"])
        db_res = await temp_db_session.execute(stmt)
        experiment_record = db_res.scalar_one_or_none()

        assert experiment_record is not None
        assert experiment_record.model_name == "LogisticRegression_Baseline"
        assert experiment_record.dataset_version == "1.0-test"

        # Декодируем и сверяем параметры и метрики из базы данных
        loaded_metrics = json.loads(experiment_record.metrics)
        assert loaded_metrics["accuracy"] == result["metrics"]["accuracy"]

@pytest.mark.asyncio
async def test_compute_baseline_holdout_f1_matches_lgbm_split(temp_db_session):
    """
    Регрессионный тест: compute_baseline_holdout_f1 должен использовать
    ровно тот же train_val/holdout split, что run_lgbm_experiment,
    и возвращать F1, посчитанный именно на holdout-сегменте.
    """
    from src.models.baseline import compute_baseline_holdout_f1, _split_train_val_holdout

    n_rows = 150
    rng = np.random.default_rng(42)
    dummy_data = {
        "open_time": range(1000, 1000 + n_rows),
        "rsi": rng.uniform(20, 80, n_rows),
        "macd": rng.normal(0, 1, n_rows),
        "macd_signal": rng.normal(0, 1, n_rows),
        "macd_hist": rng.normal(0, 1, n_rows),
        "volatility": rng.uniform(0.001, 0.05, n_rows),
        "volume_ratio": rng.uniform(0.5, 2.0, n_rows),
        "target_binary": rng.integers(0, 2, n_rows).astype(float),
    }
    df = pd.DataFrame(dummy_data)

    metadata = {
        "version": "test_v1",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "features": ["rsi", "macd", "macd_signal", "macd_hist", "volatility", "volume_ratio"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "test_dataset.parquet")
        metadata_path = os.path.join(tmpdir, "test_metadata.json")
        df.to_parquet(dataset_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        result = await compute_baseline_holdout_f1(
            temp_db_session,
            dataset_path,
            metadata_path,
            train_size=50,
            test_size=20,
        )

        assert result["f1"] is not None
        assert 0.0 <= result["f1"] <= 1.0
        assert result["holdout_size"] > 0

        # Сверяем, что holdout_size совпадает с ручным вызовом сплиттера
        feature_cols = metadata["features"]
        df_clean = df.dropna(subset=feature_cols + ["target_binary"]).reset_index(drop=True)
        _, df_holdout = _split_train_val_holdout(df_clean, train_size=50, test_size=20)
        assert result["holdout_size"] == len(df_holdout)