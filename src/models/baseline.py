import json
import asyncio
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models.backtest import TimeSeriesWalkForwardSplitter, purge_train_tail
from src.crud.experiment import ExperimentRepository
from src.datasets.build import get_git_sha
from src.core.config import get_settings
from src.labels.generator import MAX_ADAPTIVE_HORIZON_CANDLES

import warnings
# Подавляем ложные предупреждения scipy-оптимизатора при обучении LogisticRegression
try:
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)
except ImportError:
    pass



def _run_baseline_folds(splitter, df_clean, feature_cols, target_col, is_multiclass, c_parameter):
    all_y_true = []
    all_y_pred = []
    fold_count = 0

    for train_df, test_df, info in splitter.split(df_clean):
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        if is_multiclass:
            y_train = y_train.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
            y_test = y_test.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
        else:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(C=c_parameter, random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fold_count += 1

    return all_y_true, all_y_pred, fold_count

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
    settings = get_settings()

    # Безопасное получение целевой переменной с защитой от None
    target_col = getattr(settings, "TARGET_COL", "target_triple") or "target_triple"

    # Защитный механизм: автоматический откат на target_binary при отсутствии target_triple
    if target_col not in df.columns and "target_binary" in df.columns:
        target_col = "target_binary"

    feature_cols = metadata["features"]

    # Удаляем строки с пропусками
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    if len(df_clean) < (train_size + test_size):
        raise ValueError(
            "Недостаточно очищенных данных для проведения Walk-Forward оценки."
        )

    # 2. Настраиваем разделитель данных
    splitter = TimeSeriesWalkForwardSplitter(
        train_size=train_size,
        test_size=test_size,
        label_horizon=MAX_ADAPTIVE_HORIZON_CANDLES,
    )

    is_multiclass = target_col == "target_triple"
    avg_method = "macro" if is_multiclass else "binary"

    all_y_true, all_y_pred, fold_count = await asyncio.to_thread(
        _run_baseline_folds,
        splitter,
        df_clean,
        feature_cols,
        target_col,
        is_multiclass,
        c_parameter,
    )

    if fold_count == 0:
        raise ValueError(
            "Разделитель не создал ни одного фолда. Увеличьте размер датасета или уменьшите окна."
        )

    # 4. Рассчитываем итоговые метрики
    metrics = {
        "accuracy": float(accuracy_score(all_y_true, all_y_pred)),
        "precision": float(precision_score(all_y_true, all_y_pred, average=avg_method, zero_division=0)),
        "recall": float(recall_score(all_y_true, all_y_pred, average=avg_method, zero_division=0)),
        "f1": float(f1_score(all_y_true, all_y_pred, average=avg_method, zero_division=0)),
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

def _split_train_val_holdout(
    df_clean: pd.DataFrame, train_size: int, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Тот же самый хронологический split 80/20 + purge, что использует
    run_lgbm_experiment в src/models/train.py. Вынесен сюда как копия
    (не рефакторинг на общий модуль), чтобы не трогать уже работающий
    и покрытый тестами train.py.
    """
    purge_rows = MAX_ADAPTIVE_HORIZON_CANDLES
    min_train_val_needed = train_size + test_size + purge_rows

    holdout_size = int(len(df_clean) * 0.2)
    if len(df_clean) - holdout_size < min_train_val_needed:
        holdout_size = len(df_clean) - min_train_val_needed
        if holdout_size < 0:
            holdout_size = 0

    split_idx = len(df_clean) - holdout_size
    df_train_val = df_clean.iloc[:split_idx].reset_index(drop=True)
    df_holdout = df_clean.iloc[split_idx:].reset_index(drop=True)

    if len(df_holdout) > 0:
        df_train_val = purge_train_tail(df_train_val, purge_rows)

    return df_train_val, df_holdout


async def compute_baseline_holdout_f1(
    session: AsyncSession,
    dataset_path: str,
    metadata_path: str,
    train_size: int = 1000,
    test_size: int = 200,
    c_parameter: float = 1.0,
) -> dict:
    """
    Считает F1 базовой модели (LogisticRegression) на ТОМ ЖЕ split'е
    train_val/holdout, что использует run_lgbm_experiment для Quality Gate.

    В отличие от run_baseline_experiment (усреднение F1 по ВСЕМ Walk-Forward
    фолдам за всю историю датасета — легкие старые периоды рынка "разбавляют"
    метрику), это честное сравнение "яблоки к яблокам": baseline и LGBM
    обучены на одном df_train_val и проверены на одном df_holdout —
    самом свежем и обычно самом сложном по дрейфу отрезке.
    """
    df = pd.read_parquet(dataset_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    settings = get_settings()
    target_col = getattr(settings, "TARGET_COL", "target_triple") or "target_triple"
    if target_col not in df.columns and "target_binary" in df.columns:
        target_col = "target_binary"

    feature_cols = metadata["features"]
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    if len(df_clean) < (train_size + test_size):
        raise ValueError("Недостаточно очищенных данных для расчета holdout-baseline.")

    df_train_val, df_holdout = _split_train_val_holdout(df_clean, train_size, test_size)

    if len(df_holdout) == 0:
        logger.warning(
            "[MLOps Baseline] Holdout пуст (слишком маленький датасет), "
            "honest baseline F1 недоступен."
        )
        return {"f1": None, "holdout_size": 0}

    is_multiclass = target_col == "target_triple"
    avg_method = "macro" if is_multiclass else "binary"

    def _fit_and_score():
        X_train = df_train_val[feature_cols]
        y_train = df_train_val[target_col]
        X_holdout = df_holdout[feature_cols]
        y_holdout = df_holdout[target_col]

        if is_multiclass:
            y_train_m = y_train.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
            y_holdout_m = y_holdout.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
        else:
            y_train_m = y_train.astype(int)
            y_holdout_m = y_holdout.astype(int)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_holdout_scaled = scaler.transform(X_holdout)

        model = LogisticRegression(C=c_parameter, random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train_m)
        y_pred = model.predict(X_holdout_scaled)

        return f1_score(y_holdout_m, y_pred, average=avg_method, zero_division=0)

    holdout_f1 = await asyncio.to_thread(_fit_and_score)

    logger.info(
        f"[MLOps Baseline] Honest holdout F1 базовой модели "
        f"(train_val→holdout, {len(df_holdout)} строк): {holdout_f1:.4f}"
    )

    return {"f1": float(holdout_f1), "holdout_size": len(df_holdout)}