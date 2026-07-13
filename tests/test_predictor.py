import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import select

from src.db.models import Experiment
from src.models.train import run_lgbm_experiment
from src.models.predictor import Predictor


@pytest.mark.asyncio
async def test_run_lgbm_and_predict_success(temp_db_session):
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
        "version": "1.0-test-lgbm",
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
        models_dir = os.path.join(tmpdir, "models")

        df.to_parquet(dataset_path, index=False)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        # 2. Запускаем эксперимент обучения LightGBM
        result = await run_lgbm_experiment(
            session=temp_db_session,
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            train_size=100,
            test_size=20,
            learning_rate=0.1,
            n_estimators=10,
            models_dir=models_dir,
        )

        assert "experiment_id" in result
        assert os.path.exists(result["model_path"])

        # Проверяем запись в БД
        stmt = select(Experiment).where(Experiment.id == result["experiment_id"])
        db_res = await temp_db_session.execute(stmt)
        record = db_res.scalar_one_or_none()
        assert record is not None
        assert record.model_name == "LightGBM_Model"

        # 3. Тестируем Predictor на новых свечах
        predictor = Predictor(result["model_path"])

        # Создаем маленькую тестовую таблицу свечей для предсказания (30 свечей достаточно для расчета RSI)
        test_candles = pd.DataFrame(
            {
                "open_time": np.arange(2000, 2030),
                "open": np.random.uniform(100, 110, 30),
                "high": np.random.uniform(110, 120, 30),
                "low": np.random.uniform(90, 100, 30),
                "close": np.random.uniform(100, 110, 30),
                "volume": np.random.uniform(1000, 5000, 30),
            }
        )

        prediction = predictor.predict(test_candles)

        # Предсказание должно быть 0 или 1 (так как это бинарный классификатор)
        assert prediction in [0, 1]