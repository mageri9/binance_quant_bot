import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models.backtest import TimeSeriesWalkForwardSplitter
from src.crud.experiment import ExperimentRepository
from src.datasets.build import get_git_sha


async def run_baseline_experiment(
    session: AsyncSession,
    dataset_path: str,
    metadata_path: str,
    train_size: int = 1000,
    test_size: int = 200,
    c_parameter: float = 1.0,
) -> dict:
    """
    Запускает эксперимент с базовой моделью (Логистическая регрессия).
    Использует Walk-Forward нарезку для честного обучения и тестирования.
    Рассчитывает общие метрики и логирует результаты в базу данных 'experiments'.
    """
    # 1. Загружаем датасет и его описание
    df = pd.read_parquet(dataset_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Определяем признаки и целевую переменную
    feature_cols = metadata["features"]
    target_col = "target_binary"

    # Удаляем строки с пропусками (в начале таблицы из-за индикаторов, и в конце из-за горизонта будущего)
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    if len(df_clean) < (train_size + test_size):
        raise ValueError(
            "Недостаточно очищенных данных для проведения Walk-Forward оценки."
        )

    # 2. Настраиваем разделитель данных
    splitter = TimeSeriesWalkForwardSplitter(train_size=train_size, test_size=test_size)

    all_y_true = []
    all_y_pred = []
    fold_count = 0

    # 3. Запускаем обучение по шагам (фолдам)
    for train_df, test_df, info in splitter.split(df_clean):
        X_train = train_df[feature_cols]
        y_train = train_df[target_col].astype(int)

        X_test = test_df[feature_cols]
        y_test = test_df[target_col].astype(int)

        # Масштабируем признаки (Логистическая регрессия очень чувствительна к разным масштабам чисел)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Обучаем модель на прошлом
        model = LogisticRegression(C=c_parameter, random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # Предсказываем результаты на будущем
        y_pred = model.predict(X_test_scaled)

        # Накапливаем правильные ответы и предсказания для итогового расчета
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fold_count += 1

    if fold_count == 0:
        raise ValueError(
            "Разделитель не создал ни одного фолда. Увеличьте размер датасета или уменьшите окна."
        )

    # 4. Рассчитываем итоговые метрики по всей цепочке тестирования
    metrics = {
        "accuracy": float(accuracy_score(all_y_true, all_y_pred)),
        "precision": float(precision_score(all_y_true, all_y_pred, zero_division=0)),
        "recall": float(recall_score(all_y_true, all_y_pred, zero_division=0)),
        "f1": float(f1_score(all_y_true, all_y_pred, zero_division=0)),
        "total_folds": fold_count,
        "total_test_samples": len(all_y_true),
    }

    parameters = {
        "model_type": "LogisticRegression",
        "C": c_parameter,
        "train_size": train_size,
        "test_size": test_size,
        "features_used": feature_cols,
    }

    # 5. Записываем результаты эксперимента в базу данных
    repo = ExperimentRepository(session)
    experiment = await repo.log_experiment(
        model_name="LogisticRegression_Baseline",
        dataset_version=metadata["version"],
        parameters=parameters,
        metrics=metrics,
        git_sha=get_git_sha(),
    )

    logger.info(
        f"Эксперимент Baseline сохранен в БД. ID: {experiment.id}. "
        f"Accuracy: {metrics['accuracy']:.4f}, F1-score: {metrics['f1']:.4f}"
    )

    return {
        "experiment_id": experiment.id,
        "parameters": parameters,
        "metrics": metrics,
    }