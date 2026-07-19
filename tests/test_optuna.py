import pytest
import pandas as pd
import numpy as np
from sqlalchemy import select

from src.db.models import Experiment
from src.models.train import tune_lgbm_hyperparameters


@pytest.mark.asyncio
async def test_optuna_tuning_success(temp_db_session):
    # Генерируем тестовый датасет на 150 строк
    np.random.seed(42)
    n_rows = 150

    dummy_data = {
        "open_time": np.arange(1000, 1000 + n_rows),
        "open": np.random.uniform(100, 110, n_rows),
        "high": np.random.uniform(110, 120, n_rows),
        "low": np.random.uniform(90, 100, n_rows),
        "close": np.random.uniform(100, 110, n_rows),
        "volume": np.random.uniform(1000, 5000, n_rows),
        "rsi": np.random.uniform(20, 80, n_rows),
        "macd": np.random.uniform(-1, 1, n_rows),
        "macd_signal": np.random.uniform(-1, 1, n_rows),
        "macd_hist": np.random.uniform(-1, 1, n_rows),
        "volatility": np.random.uniform(0.01, 0.05, n_rows),
        "volume_ratio": np.random.uniform(0.5, 2.0, n_rows),
        "target_binary": np.random.choice([0.0, 1.0], size=n_rows),
    }
    df = pd.DataFrame(dummy_data)

    feature_cols = ["rsi", "macd", "macd_signal", "macd_hist", "volatility", "volume_ratio"]

    # Запускаем подбор параметров с n_trials=2 для скорости прохождения тестов
    best_params = await tune_lgbm_hyperparameters(
        session=temp_db_session,
        df_clean=df,
        feature_cols=feature_cols,
        target_col="target_binary",
        train_size=100,
        test_size=20,
        metadata_version="test-tuning",
        n_trials=2,
    )

    # 1. Проверяем, что параметры успешно подобраны и возвращены словарем
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert "num_leaves" in best_params

    # 2. Проверяем, что в БД сохранилась запись об этом эксперименте тюнинга
    stmt = select(Experiment).where(Experiment.model_name == "LightGBM_Hyperparameter_Tuning")
    db_res = await temp_db_session.execute(stmt)
    tuning_record = db_res.scalar_one_or_none()

    assert tuning_record is not None
    assert tuning_record.dataset_version == "test-tuning"


@pytest.mark.asyncio
async def test_optuna_tuning_sharpe_objective(temp_db_session):
    np.random.seed(42)
    n_rows = 150

    dummy_data = {
        "open_time": np.arange(1000, 1000 + n_rows),
        "open": np.random.uniform(100, 110, n_rows),
        "high": np.random.uniform(110, 120, n_rows),
        "low": np.random.uniform(90, 100, n_rows),
        "close": np.random.uniform(100, 110, n_rows),
        "volume": np.random.uniform(1000, 5000, n_rows),
        "rsi": np.random.uniform(20, 80, n_rows),
        "macd": np.random.uniform(-1, 1, n_rows),
        "macd_signal": np.random.uniform(-1, 1, n_rows),
        "macd_hist": np.random.uniform(-1, 1, n_rows),
        "volatility": np.random.uniform(0.01, 0.05, n_rows),
        "volume_ratio": np.random.uniform(0.5, 2.0, n_rows),
        "target_binary": np.random.choice([0.0, 1.0], size=n_rows),
    }
    df = pd.DataFrame(dummy_data)
    feature_cols = ["rsi", "macd", "macd_signal", "macd_hist", "volatility", "volume_ratio"]

    best_params = await tune_lgbm_hyperparameters(
        session=temp_db_session,
        df_clean=df,
        feature_cols=feature_cols,
        target_col="target_binary",
        train_size=100,
        test_size=20,
        metadata_version="test-tuning-sharpe",
        n_trials=2,
        objective_metric="sharpe",
    )

    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params

    stmt = select(Experiment).where(Experiment.model_name == "LightGBM_Hyperparameter_Tuning")
    db_res = await temp_db_session.execute(stmt)
    records = db_res.scalars().all()
    tuning_record = next(r for r in records if r.dataset_version == "test-tuning-sharpe")

    assert tuning_record is not None


@pytest.mark.asyncio
async def test_optuna_tuning_rejects_invalid_objective_metric(temp_db_session):
    df = pd.DataFrame({
        "open_time": [1, 2, 3],
        "close": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "rsi": [50.0, 51.0, 52.0],
        "target_binary": [1.0, 0.0, 1.0],
    })
    with pytest.raises(ValueError):
        await tune_lgbm_hyperparameters(
            session=temp_db_session,
            df_clean=df,
            feature_cols=["rsi"],
            target_col="target_binary",
            train_size=1,
            test_size=1,
            metadata_version="x",
            n_trials=1,
            objective_metric="bogus",
        )