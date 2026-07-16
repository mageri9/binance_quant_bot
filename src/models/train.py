import os
import json
import asyncio
import pickle
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models.backtest import TimeSeriesWalkForwardSplitter, purge_train_tail
from src.crud.experiment import ExperimentRepository
from src.datasets.build import get_git_sha
from src.core.config import get_settings
from src.labels.generator import MAX_ADAPTIVE_HORIZON_CANDLES
from datetime import datetime, timezone

from src.utils.artifact_paths import get_oos_path

# Отключаем избыточный вывод логов Optuna в консоль
optuna.logging.set_verbosity(optuna.logging.WARNING)


async def tune_lgbm_hyperparameters(
    session: AsyncSession,
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_size: int,
    test_size: int,
    metadata_version: str,
    n_trials: int = 15,
    label_horizon: int = 0,
) -> dict:
    """
    Проводит автоматический подбор параметров LightGBM с помощью Optuna.
    Минимизирует или максимизирует средний F1-score по фолдам Walk-Forward.
    """
    is_multiclass = target_col == "target_triple"
    avg_method = "macro" if is_multiclass else "binary"

    def objective(trial):
        # Задаем пространство поиска параметров
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 1,
        }

        splitter = TimeSeriesWalkForwardSplitter(
            train_size=train_size,
            test_size=test_size,
            label_horizon=label_horizon,
        )
        f1_scores = []

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

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
            f1_scores.append(score)

        if not f1_scores:
            return 0.0

        return float(np.mean(f1_scores))

    # Запускаем оптимизацию
    study = optuna.create_study(direction="maximize")
    await asyncio.to_thread(study.optimize, objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    # Записываем результаты поиска параметров в БД экспериментов
    repo = ExperimentRepository(session)
    tuning_parameters = {
        "model_type": "LightGBM_Optuna_Tuning",
        "search_space": {
            "learning_rate": [0.01, 0.2],
            "n_estimators": [50, 300],
            "max_depth": [3, 10],
            "num_leaves": [10, 100],
        },
        "best_params": best_params,
        "n_trials": n_trials,
        "features_used": feature_cols,
    }
    tuning_metrics = {
        "best_cv_f1_score": best_value,
        "n_trials_completed": n_trials,
    }

    await repo.log_experiment(
        model_name="LightGBM_Hyperparameter_Tuning",
        dataset_version=metadata_version,
        parameters=tuning_parameters,
        metrics=tuning_metrics,
        git_sha=get_git_sha(),
    )

    return best_params


def _run_walk_forward_folds(
    df_train_val, feature_cols, target_col, train_size, test_size,
    label_horizon, model_kwargs, is_multiclass, avg_method,
):
    splitter = TimeSeriesWalkForwardSplitter(
        train_size=train_size, test_size=test_size, label_horizon=label_horizon,
    )

    all_y_true = []
    all_y_pred = []
    fold_count = 0
    best_model = None
    best_fold_f1 = -float("inf")
    oos_dfs = []

    for train_df, test_df, info in splitter.split(df_train_val):
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

        model = LGBMClassifier(**model_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        fold_f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        test_df_copy = test_df.copy()
        if is_multiclass:
            signal_map = {0: -1.0, 1: 0.0, 2: 1.0}
            test_df_copy["predicted_signal"] = pd.Series(y_pred, index=test_df.index).map(signal_map)
        else:
            test_df_copy["predicted_signal"] = y_pred

        oos_dfs.append(test_df_copy)

        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_model = model

        fold_count += 1

    return all_y_true, all_y_pred, fold_count, best_model, best_fold_f1, oos_dfs

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
    baseline_f1: float
    | None = None,  # ← Передаем F1-score базовой модели для Quality Gate
    bypass_quality_gates: bool = False,  # ← Защитный флаг для юнит-тестов на случайных данных
) -> dict:
    """
    Запускает эксперимент с моделью LightGBM с hold-out валидацией и Quality Gates.
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
        raise ValueError(
            "Недостаточно очищенных данных для проведения Walk-Forward оценки."
        )

    # --- 1. СТРОГИЙ ХРОНОЛОГИЧЕСКИЙ SPLIT (80% / 20%) ---
    purge_rows = MAX_ADAPTIVE_HORIZON_CANDLES
    min_train_val_needed = train_size + test_size + purge_rows

    holdout_size = int(len(df_clean) * 0.2)
    # Для крошечных тест-выборок гарантируем, что останется хотя бы один фолд
    if len(df_clean) - holdout_size < min_train_val_needed:
        holdout_size = len(df_clean) - min_train_val_needed
        if holdout_size < 0:
            holdout_size = 0

    split_idx = len(df_clean) - holdout_size
    df_train_val = df_clean.iloc[:split_idx].reset_index(drop=True)
    df_holdout = df_clean.iloc[split_idx:].reset_index(drop=True)

    if len(df_holdout) > 0:
        df_train_val = purge_train_tail(df_train_val, purge_rows)

    logger.info(
        f"[MLOps Train] Всего строк: {len(df_clean)}. Валидация: {len(df_train_val)}, "
        f"Holdout: {len(df_holdout)}, Purge: {purge_rows if len(df_holdout) > 0 else 0}"
    )

    # 1.5. Шаг автоматической калибровки параметров через Optuna (если включено в конфиге)
    best_params = {}
    if settings.OPTUNA_TUNING_ENABLED:
        logger.info(
            f"[*] Запуск подбора гиперпараметров через Optuna ({settings.OPTUNA_TRIALS} попыток)..."
        )
        best_params = await tune_lgbm_hyperparameters(
            session=session,
            df_clean=df_train_val,
            feature_cols=feature_cols,
            target_col=target_col,
            train_size=train_size,
            test_size=test_size,
            metadata_version=metadata["version"],
            n_trials=settings.OPTUNA_TRIALS,
            label_horizon=purge_rows,
        )
        logger.info(f"[+] Лучшие параметры подобраны: {best_params}")

    is_multiclass = target_col == "target_triple"
    avg_method = "macro" if is_multiclass else "binary"

    model_kwargs = {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": 1,
    }
    if settings.OPTUNA_TUNING_ENABLED and best_params:
        model_kwargs.update(best_params)

    (
        all_y_true,
        all_y_pred,
        fold_count,
        best_model,
        best_fold_f1,
        oos_dfs,
    ) = await asyncio.to_thread(
        _run_walk_forward_folds,
        df_train_val,
        feature_cols,
        target_col,
        train_size,
        test_size,
        purge_rows,
        model_kwargs,
        is_multiclass,
        avg_method,
    )

    if fold_count == 0 or best_model is None:
        raise ValueError("Не удалось запустить Walk-Forward. Проверьте размер данных.")

    # --- 3. СТРОГИЙ ТЕСТ НА HOLDOUT (QUALITY GATES) ---
    if len(df_holdout) > 0 and not bypass_quality_gates:
        X_holdout = df_holdout[feature_cols]
        y_holdout = df_holdout[target_col]

        if is_multiclass:
            y_holdout_mapped = y_holdout.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
        else:
            y_holdout_mapped = y_holdout.astype(int)

        y_holdout_pred = best_model.predict(X_holdout)
        holdout_f1 = f1_score(
            y_holdout_mapped, y_holdout_pred, average=avg_method, zero_division=0
        )

        # Quality Gate 1: Сравнение с Baseline (F1-score на holdout должен быть выше базовой модели)
        if baseline_f1 is not None and holdout_f1 <= baseline_f1:
            raise ValueError(
                f"LGBM model REJECTED by Quality Gate: "
                f"Holdout F1 ({holdout_f1:.4f}) does not exceed Baseline F1 ({baseline_f1:.4f})"
            )

        # Quality Gate 2: Защита от вырождения предсказаний (Class Collapse Protection)
        unique_preds, counts = np.unique(y_holdout_pred, return_counts=True)
        pred_ratios = counts / len(y_holdout_pred)
        for val, ratio in zip(unique_preds, pred_ratios):
            if ratio >= 0.95:
                raise ValueError(
                    f"LGBM model REJECTED by Quality Gate: "
                    f"Class collapse detected. Class {val} occupies {ratio:.1%} of predictions on hold_out."
                )
        logger.info(
            f"[MLOps Train] Модель успешно прошла все Quality Gates на Holdout. Holdout F1: {holdout_f1:.4f}"
        )

    # --- 4. ФИНАЛЬНОЕ ДООБУЧЕНИЕ НА 100% ДАННЫХ (FIT ON ALL) ---
    logger.info(
        "[MLOps Train] Запуск финального дообучения модели на 100% исторических данных..."
    )
    final_model = LGBMClassifier(**model_kwargs)

    X_all = df_clean[feature_cols]
    y_all = df_clean[target_col]
    if is_multiclass:
        y_all_mapped = y_all.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    else:
        y_all_mapped = y_all.astype(int)

    await asyncio.to_thread(final_model.fit, X_all, y_all_mapped)

    # Объединяем OOS фолды
    df_oos = pd.concat(oos_dfs).sort_values("open_time").reset_index(drop=True)

    # Объединяем OOS фолды
    df_oos = pd.concat(oos_dfs).sort_values("open_time").reset_index(drop=True)

    # 5. Метрики для записи в БД экспериментов
    metrics = {
        "accuracy": float(accuracy_score(all_y_true, all_y_pred)),
        "precision": float(
            precision_score(all_y_true, all_y_pred, average=avg_method, zero_division=0)
        ),
        "recall": float(
            recall_score(all_y_true, all_y_pred, average=avg_method, zero_division=0)
        ),
        "f1": float(
            f1_score(all_y_true, all_y_pred, average=avg_method, zero_division=0)
        ),
        "total_folds": fold_count,
        "total_test_samples": len(all_y_true),
    }

    parameters = {
        "model_type": "LightGBM",
        "learning_rate": best_params.get("learning_rate", learning_rate)
        if best_params
        else learning_rate,
        "n_estimators": best_params.get("n_estimators", n_estimators)
        if best_params
        else n_estimators,
        "max_depth": best_params.get("max_depth", max_depth)
        if best_params
        else max_depth,
        "train_size": train_size,
        "test_size": test_size,
        "features_used": feature_cols,
    }
    if best_params.get("num_leaves"):
        parameters["num_leaves"] = best_params["num_leaves"]

    repo = ExperimentRepository(session)
    experiment = await repo.log_experiment(
        model_name="LightGBM_Model",
        dataset_version=metadata["version"],
        parameters=parameters,
        metrics=metrics,
        git_sha=get_git_sha(),
    )

    # 6. Сохранение упакованного ModelArtifact
    os.makedirs(models_dir, exist_ok=True)
    clean_symbol = metadata["symbol"].replace("/", "").replace(":", "")
    clean_tf = metadata["timeframe"].replace("/", "")
    model_filename = f"lgbm_{clean_symbol}_{clean_tf}.pkl"
    model_path = os.path.join(models_dir, model_filename)

    import hashlib

    features_str = ",".join(sorted(feature_cols))
    features_hash = hashlib.sha256(features_str.encode("utf-8")).hexdigest()[:12]

    artifact = {
        "model_id": f"lgbm_{clean_symbol}_{clean_tf}_{metadata['version']}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": metadata["symbol"],
        "timeframe": metadata["timeframe"],
        "dataset_version": metadata["version"],
        "git_sha": get_git_sha(),
        "target_col": target_col,
        "features": feature_cols,
        "features_hash": features_hash,
        "model": final_model,
        "scaler": None,
        "calibration": {
            "sl_pct": settings.PAPER_SL_PCT,
            "tp_pct": settings.PAPER_TP_PCT,
            "horizon": settings.LABEL_HORIZON,
            "sharpe_ratio": None,
            "calibrated_at": None,
        },
        "metrics": metrics,
        # df_oos больше не хранится внутри pickle — см. get_oos_path().
        # Каждая загрузка модели (инференс, откат) больше не тащит в память
        # весь OOS-датафрейм, который нужен только для калибровки/drift-проверки.
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    # Сохраняем OOS отдельно, рядом с моделью — читается точечно только там,
    # где реально нужен (scripts/calibrate.py, drift-детекция в src/main.py).
    oos_path = get_oos_path(model_path)
    await asyncio.to_thread(df_oos.to_parquet, oos_path, index=False)

    return {
        "experiment_id": experiment.id,
        "model_path": model_path,
        "parameters": parameters,
        "metrics": metrics,
    }