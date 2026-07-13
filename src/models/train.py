import os
import json
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models.backtest import TimeSeriesWalkForwardSplitter
from src.crud.experiment import ExperimentRepository
from src.datasets.build import get_git_sha


async def run_lgbm_experiment(
    session: AsyncSession,
    dataset_path: str,
    metadata_path: str,
    train_size: int = 1000,
    test_size: int = 200,
    learning_rate: float = 0.05,
    n_estimators: int = 100,
    max_depth: int = -1,
    models_dir: str = "models/saved_models",
) -> dict:
    """
    Запускает эксперимент с продвинутой моделью LightGBM.
    Обучает модель на Walk-Forward нарезках и записывает результаты в 'experiments'.
    Также сохраняет финальную обученную модель в файл для использования в Predictor.
    """
    # 1. Загружаем датасет и его описание
    df = pd.read_parquet(dataset_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["features"]
    target_col = "target_binary"

    # Удаляем строки с пустыми значениями
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    if len(df_clean) < (train_size + test_size):
        raise ValueError(
            "Недостаточно очищенных данных для проведения Walk-Forward оценки."
        )

    splitter = TimeSeriesWalkForwardSplitter(train_size=train_size, test_size=test_size)

    all_y_true = []
    all_y_pred = []
    fold_count = 0

    # Сюда сохраним модель на самом последнем шаге как наиболее актуальную
    final_model = None

    # 2. Walk-Forward цикл обучения
    for train_df, test_df, info in splitter.split(df_clean):
        X_train = train_df[feature_cols]
        y_train = train_df[target_col].astype(int)

        X_test = test_df[feature_cols]
        y_test = test_df[target_col].astype(int)

        # LightGBM не требует масштабирования признаков, обучаем напрямую
        model = LGBMClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            verbosity=-1,  # Отключаем лишний вывод логов в консоль
        )
        model.fit(X_train, y_train)

        # Делаем предсказание
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fold_count += 1
        final_model = model

    if fold_count == 0:
        raise ValueError("Не удалось запустить Walk-Forward. Проверьте размер данных.")

    # 3. Рассчитываем метрики точности
    metrics = {
        "accuracy": float(accuracy_score(all_y_true, all_y_pred)),
        "precision": float(precision_score(all_y_true, all_y_pred, zero_division=0)),
        "recall": float(recall_score(all_y_true, all_y_pred, zero_division=0)),
        "f1": float(f1_score(all_y_true, all_y_pred, zero_division=0)),
        "total_folds": fold_count,
        "total_test_samples": len(all_y_true),
    }

    parameters = {
        "model_type": "LightGBM",
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "train_size": train_size,
        "test_size": test_size,
        "features_used": feature_cols,
    }

    # 4. Сохраняем модель в БД экспериментов
    repo = ExperimentRepository(session)
    experiment = await repo.log_experiment(
        model_name="LightGBM_Model",
        dataset_version=metadata["version"],
        parameters=parameters,
        metrics=metrics,
        git_sha=get_git_sha(),
    )

    # 5. Сохраняем файл модели на диск для использования в Predictor
    os.makedirs(models_dir, exist_ok=True)
    clean_symbol = metadata["symbol"].replace("/", "").replace(":", "")
    model_filename = f"lgbm_{clean_symbol}_{metadata['timeframe'].replace('/', '')}.pkl"
    model_path = os.path.join(models_dir, model_filename)

    saved_data = {
        "model": final_model,
        "features": feature_cols,
        "scaler": None,  # LightGBM не использует StandardScaler
        "symbol": metadata["symbol"],
        "timeframe": metadata["timeframe"],
        "version": metadata["version"],
    }

    with open(model_path, "wb") as f:
        pickle.dump(saved_data, f)

    logger.info(
        f"Модель LightGBM успешно обучена и сохранена в {model_path}. "
        f"Accuracy: {metrics['accuracy']:.4f}, F1-score: {metrics['f1']:.4f}"
    )

    return {
        "experiment_id": experiment.id,
        "model_path": model_path,
        "parameters": parameters,
        "metrics": metrics,
    }