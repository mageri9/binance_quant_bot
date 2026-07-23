import os
import json
import pickle
import tempfile
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import select

from src.db.models import Experiment
from src.models.train import run_lgbm_experiment
from src.models.predictor import Predictor
from src.utils.artifact_paths import get_oos_path


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
            bypass_quality_gates=True,  # ← Исключаем случайные падения тестов на шуме
        )

        oos_path = get_oos_path(result["model_path"])
        assert os.path.exists(oos_path)

        with open(result["model_path"], "rb") as f:
            raw_artifact = pickle.load(f)
        assert "df_oos" not in raw_artifact

        df_oos_loaded = pd.read_parquet(oos_path)
        assert len(df_oos_loaded) > 0
        assert "predicted_signal" in df_oos_loaded.columns

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

class _FakePrimaryModel:
    classes_ = [0, 1]
    def predict(self, X):
        return np.array([1])
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


class _FakeMetaModel:
    classes_ = [0, 1]
    def __init__(self, success_prob):
        self.success_prob = success_prob
    def predict_proba(self, X):
        return np.array([[1 - self.success_prob, self.success_prob]])


def _build_artifact(meta_model, meta_features):
    return {
        "model": _FakePrimaryModel(),
        "scaler": None,
        "features": ["rsi", "macd", "macd_signal", "macd_hist", "volatility", "volume_ratio"],
        "target_col": "target_binary",
        "meta_model": meta_model,
        "meta_features": meta_features,
    }


def _make_test_candles():
    np.random.seed(0)
    return pd.DataFrame({
        "open_time": np.arange(2000, 2030),
        "open": np.random.uniform(100, 110, 30),
        "high": np.random.uniform(110, 120, 30),
        "low": np.random.uniform(90, 100, 30),
        "close": np.random.uniform(100, 110, 30),
        "volume": np.random.uniform(1000, 5000, 30),
    })


def test_predictor_meta_model_gates_low_confidence_signal(tmp_path):
    artifact = _build_artifact(_FakeMetaModel(success_prob=0.1), ["adx", "atr_pct", "predicted_signal"])
    model_path = str(tmp_path / "fake_meta_low.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    predictor = Predictor(model_path, confidence_threshold=0.5, meta_threshold=0.5)
    assert predictor.predict(_make_test_candles()) == 0


def test_predictor_meta_model_allows_high_confidence_signal(tmp_path):
    artifact = _build_artifact(_FakeMetaModel(success_prob=0.9), ["adx", "atr_pct", "predicted_signal"])
    model_path = str(tmp_path / "fake_meta_high.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    predictor = Predictor(model_path, confidence_threshold=0.5, meta_threshold=0.5)
    assert predictor.predict(_make_test_candles()) == 1


def test_predictor_without_meta_model_unaffected(tmp_path):
    artifact = _build_artifact(None, None)
    model_path = str(tmp_path / "fake_no_meta.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    predictor = Predictor(model_path, confidence_threshold=0.5)
    assert predictor.predict(_make_test_candles()) == 1


def test_predictor_uses_calibrated_artifact_edge_threshold(tmp_path):
    artifact = _build_artifact(None, None)
    artifact["edge_threshold"] = 0.95
    model_path = str(tmp_path / "fake_edge_threshold.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    predictor = Predictor(model_path)
    assert predictor.predict(_make_test_candles()) == 0

class _FakeMetaModelWithDrift:
    classes_ = [0, 1]
    def predict_proba(self, X):
        assert "regime_drift_pvalue" in X.columns
        return np.array([[0.1, 0.9]])


class _FakeMetaModelRejectAll:
    classes_ = [0, 1]
    def predict_proba(self, X):
        return np.array([[0.9, 0.1]])


def test_predictor_meta_model_uses_stored_drift_pvalue(tmp_path):
    artifact = _build_artifact(
        _FakeMetaModelWithDrift(), ["adx", "regime_drift_pvalue", "predicted_signal"],
    )
    artifact["regime_drift_pvalue_at_train"] = 0.02
    model_path = str(tmp_path / "fake_meta_drift.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    predictor = Predictor(model_path, confidence_threshold=0.5, meta_threshold=0.5)
    assert predictor.predict(_make_test_candles()) == 1


def test_predictor_meta_model_skips_gate_when_drift_missing(tmp_path):
    """Если meta-модель требует drift-признак, а он не был сохранён (старый артефакт до 6c) —
    гейт молча пропускается (meta_ok=False), сигнал первичной модели проходит без фильтра."""
    artifact = _build_artifact(
        _FakeMetaModelRejectAll(), ["adx", "regime_drift_pvalue", "predicted_signal"],
    )
    # regime_drift_pvalue_at_train намеренно не задан
    model_path = str(tmp_path / "fake_meta_no_drift.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    predictor = Predictor(model_path, confidence_threshold=0.5, meta_threshold=0.5)
    assert predictor.predict(_make_test_candles()) == 1  # гейт пропущен, сигнал не погашен
