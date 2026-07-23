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
from src.strategy.signals import simulate_strategy
from src.strategy.edge import apply_edge_threshold, sweep_edge_thresholds
from src.execution.kernel import ExecutionKernel, costs_from_settings

from src.utils.artifact_paths import get_oos_path

# Отключаем избыточный вывод логов Optuna в консоль
optuna.logging.set_verbosity(optuna.logging.WARNING)

SIGNAL_MAP_TRIPLE = {0: -1.0, 1: 0.0, 2: 1.0}
WFO_SPLIT_COLUMN = "wfo_split"


def split_nested_wfo_data(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    purge_rows: int,
    calibration_fraction: float,
    economic_test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological inner-train, calibration, and untouched test sets.

    Hyperparameter selection may only inspect ``inner_train``.  The calibration
    segment selects trading-risk parameters, while the final segment remains
    unavailable until those choices have been made.
    """
    min_inner_rows = train_size + test_size + purge_rows
    if len(df) < min_inner_rows + 2:
        raise ValueError("Not enough rows for nested WFO train/calibration/test split.")

    available_holdout = len(df) - min_inner_rows
    requested_economic_test = max(test_size, int(len(df) * economic_test_fraction))
    economic_test_size = min(requested_economic_test, available_holdout - 1)
    requested_calibration = max(test_size, int((len(df) - economic_test_size) * calibration_fraction))
    calibration_size = min(requested_calibration, available_holdout - economic_test_size)

    if calibration_size < 1 or economic_test_size < 1:
        raise ValueError("Nested WFO requires non-empty calibration and economic test sets.")

    inner_end = len(df) - calibration_size - economic_test_size
    calibration_end = len(df) - economic_test_size
    inner_train = purge_train_tail(df.iloc[:inner_end].copy(), purge_rows)
    calibration = df.iloc[inner_end:calibration_end].copy()
    economic_test = df.iloc[calibration_end:].copy()

    if len(inner_train) < train_size + test_size:
        raise ValueError("Nested WFO inner-train is too short after purging labels.")
    return inner_train.reset_index(drop=True), calibration.reset_index(drop=True), economic_test.reset_index(drop=True)

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
    max_folds: int | None = None,
    objective_metric: str = "f1",
    symbol: str = "unknown",
    timeframe: str = "unknown",
    schema_hash: str = "unknown",
) -> dict:
    """
    Проводит автоматический подбор параметров LightGBM с помощью Optuna.

    objective_metric: "f1" (по умолчанию) — максимизирует средний F1-score
    по фолдам Walk-Forward, как раньше. "sharpe" / "expectancy" — вместо
    этого на каждом фолде строит predicted_signal из y_pred и запускает
    simulate_strategy на test_df фолда (там уже есть close/high/low),
    максимизируя средний Sharpe/Expectancy по фолдам. Требует наличия
    колонок close/high/low в df_clean.
    """
    is_multiclass = target_col == "target_triple"
    avg_method = "macro" if is_multiclass else "binary"

    if objective_metric not in ("f1", "sharpe", "expectancy"):
        raise ValueError(f"Неизвестный objective_metric: {objective_metric}")

    def _fold_score(y_test, y_pred, test_df):
        if objective_metric == "f1":
            return f1_score(y_test, y_pred, average=avg_method, zero_division=0)

        if is_multiclass:
            predicted_signal = pd.Series(y_pred, index=test_df.index).map(SIGNAL_MAP_TRIPLE)
        else:
            # Бинарная модель: 1 -> LONG, 0 -> HOLD (нет отдельного шорт-класса)
            predicted_signal = pd.Series(y_pred, index=test_df.index).map({0: 0.0, 1: 1.0})

        sim_df = test_df.copy()
        sim_df["predicted_signal"] = predicted_signal

        settings = get_settings()
        sim_metrics = simulate_strategy(
            sim_df,
            predicted_col="predicted_signal",
            execution_kernel=ExecutionKernel(costs_from_settings(settings)),
        )
        if sim_metrics["total_trades"] < settings.OPTUNA_MIN_TRADES:
            return -1e6
        return sim_metrics["sharpe_ci_low"] if objective_metric == "sharpe" else sim_metrics["expectancy"]

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 1,
            "class_weight": "balanced",
        }

        splitter = TimeSeriesWalkForwardSplitter(
            train_size=train_size,
            test_size=test_size,
            label_horizon=label_horizon,
        )
        fold_scores = []

        folds = list(splitter.split(df_clean))
        if max_folds is not None and len(folds) > max_folds:
            folds = folds[-max_folds:]

        for train_df, test_df, info in folds:
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
            fold_scores.append(_fold_score(y_test, y_pred, test_df))

        if not fold_scores:
            return 0.0

        return float(np.mean(fold_scores))

    settings = get_settings()
    storage_url = settings.OPTUNA_STORAGE_URL or None
    study_name = "lgbm__" + "__".join(
        value.replace("/", "_").replace(":", "_")
        for value in (symbol, timeframe, target_col, schema_hash)
    )
    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=settings.OPTUNA_SEED),
    )
    await asyncio.to_thread(study.optimize, objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

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
        "objective_metric": objective_metric,
        "study_name": study_name,
        "storage_persistent": storage_url is not None,
    }
    tuning_metrics = {
        f"best_cv_{objective_metric}_score": best_value,
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
        y_proba = model.predict_proba(X_test)
        confidence = y_proba.max(axis=1)
        fold_f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        test_df_copy = test_df.copy()
        if is_multiclass:
            signal_map = {0: -1.0, 1: 0.0, 2: 1.0}
            test_df_copy["predicted_signal"] = pd.Series(
                y_pred, index=test_df.index
            ).map(signal_map)
        else:
            test_df_copy["predicted_signal"] = y_pred
        test_df_copy["predicted_confidence"] = confidence

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
    baseline_f1: float | None = None,
    regime_drift_pvalue: float | None = None,
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
    df_inner_train, df_calibration, df_economic_test = split_nested_wfo_data(
        df_clean,
        train_size=train_size,
        test_size=test_size,
        purge_rows=purge_rows,
        calibration_fraction=settings.WFO_CALIBRATION_FRACTION,
        economic_test_fraction=settings.WFO_ECONOMIC_TEST_FRACTION,
    )
    # Keep legacy local names below while enforcing the nested chronological split.
    df_train_val = df_inner_train
    df_holdout = df_economic_test
    logger.info(
        f"[MLOps Train] Nested WFO rows: inner_train={len(df_inner_train)}, "
        f"calibration={len(df_calibration)}, economic_test={len(df_economic_test)}, purge={purge_rows}"
    )

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
            max_folds=settings.OPTUNA_MAX_FOLDS,
            objective_metric=getattr(settings, "OPTUNA_OBJECTIVE_METRIC", "f1"),
            symbol=metadata["symbol"],
            timeframe=metadata["timeframe"],
            schema_hash=metadata.get("feature_schema_hash", "legacy"),
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
        "class_weight": "balanced",
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

    holdout_f1 = None

        # --- 3. СТРОГИЙ ТЕСТ НА HOLDOUT (QUALITY GATES) ---
        # ВАЖНО: гейт проверяет НЕ best_model (лучший случайный фолд Walk-Forward —
        # переобучение на удачном окне рынка) и НЕ final_model (она уже видела holdout
        # при обучении на 100% данных — это была бы утечка). Гейт проверяет отдельного
        # кандидата, обученного ровно на df_train_val (без holdout), максимально
        # близкого по объему обучающих данных к тому, что реально уедет в прод.
    if len(df_holdout) > 0 and not bypass_quality_gates:
        logger.info(
            "[MLOps Train] Обучаю gate-кандидата на df_train_val (без holdout данных)..."
        )
        gate_candidate = LGBMClassifier(**model_kwargs)
        X_train_val = df_train_val[feature_cols]
        y_train_val = df_train_val[target_col]
        if is_multiclass:
            y_train_val_mapped = y_train_val.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
        else:
            y_train_val_mapped = y_train_val.astype(int)
        await asyncio.to_thread(gate_candidate.fit, X_train_val, y_train_val_mapped)

        X_holdout = df_holdout[feature_cols]
        y_holdout = df_holdout[target_col]

        if is_multiclass:
            y_holdout_mapped = y_holdout.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
        else:
            y_holdout_mapped = y_holdout.astype(int)

        y_holdout_pred = gate_candidate.predict(X_holdout)
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
    # Produce downstream predictions from an inner-train-only candidate. The
    # economic segment is never used while selecting risk parameters.
    partition_candidate = LGBMClassifier(**model_kwargs)
    y_inner = df_inner_train[target_col]
    if is_multiclass:
        y_inner = y_inner.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    else:
        y_inner = y_inner.astype(int)
    await asyncio.to_thread(partition_candidate.fit, df_inner_train[feature_cols], y_inner)

    def _prediction_frame(source_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        result = source_df.copy()
        predicted = partition_candidate.predict(result[feature_cols])
        probabilities = partition_candidate.predict_proba(result[feature_cols])
        if is_multiclass:
            result["predicted_signal"] = pd.Series(predicted, index=result.index).map(SIGNAL_MAP_TRIPLE)
        else:
            result["predicted_signal"] = predicted
        result["predicted_confidence"] = probabilities.max(axis=1)
        result[WFO_SPLIT_COLUMN] = split_name
        return result

    calibration_oos = _prediction_frame(df_calibration, "calibration")
    economic_test_oos = _prediction_frame(df_economic_test, "economic_test")

    edge_threshold = settings.PREDICTION_CONFIDENCE_THRESHOLD
    edge_sweep = []
    if settings.EDGE_THRESHOLD_SWEEP_ENABLED:
        edge_threshold, edge_sweep = await asyncio.to_thread(
            sweep_edge_thresholds,
            calibration_oos,
            settings.EDGE_THRESHOLD_GRID,
            settings.EDGE_MIN_COVERAGE,
            settings.CALIBRATION_MIN_TRADES,
            {
                "horizon": settings.LABEL_HORIZON,
                "sl_pct": settings.PAPER_SL_PCT,
                "tp_pct": settings.PAPER_TP_PCT,
                "execution_kernel": ExecutionKernel(costs_from_settings(settings)),
            },
        )
        logger.info(
            f"[Edge sweep] selected confidence threshold={edge_threshold:.2f} "
            f"from {len(edge_sweep)} calibration candidates"
        )

    calibration_oos = apply_edge_threshold(calibration_oos, edge_threshold)
    economic_test_oos = apply_edge_threshold(economic_test_oos, edge_threshold)
    economic_backtest_metrics = simulate_strategy(
        economic_test_oos,
        horizon=settings.LABEL_HORIZON,
        sl_pct=settings.PAPER_SL_PCT,
        tp_pct=settings.PAPER_TP_PCT,
        execution_kernel=ExecutionKernel(costs_from_settings(settings)),
    )

    final_model = LGBMClassifier(**model_kwargs)

    df_final_train = pd.concat([df_inner_train, df_calibration], ignore_index=True)
    X_all = df_final_train[feature_cols]
    y_all = df_final_train[target_col]
    if is_multiclass:
        y_all_mapped = y_all.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    else:
        y_all_mapped = y_all.astype(int)

    await asyncio.to_thread(final_model.fit, X_all, y_all_mapped)

    # Объединяем OOS фолды
    inner_oos = pd.concat(oos_dfs).sort_values("open_time").reset_index(drop=True)
    inner_oos[WFO_SPLIT_COLUMN] = "inner_wfo"
    df_oos = pd.concat([inner_oos, calibration_oos, economic_test_oos]).sort_values("open_time").reset_index(drop=True)

    meta_model = None
    meta_feature_cols = None
    meta_metrics = {}
    if getattr(settings, "META_LABELING_ENABLED", False):
        try:
            from src.models.meta import (
                build_meta_dataset,
                train_meta_model,
                META_BASE_FEATURES,
            )

            meta_df = await asyncio.to_thread(
                build_meta_dataset,
                inner_oos,
                drift_pvalue=regime_drift_pvalue,
            )
            candidate_features = list(META_BASE_FEATURES)
            if regime_drift_pvalue is not None:
                candidate_features.append("regime_drift_pvalue")
            candidate_features += [
                c for c in ("predicted_signal", "predicted_confidence") if c in meta_df.columns
            ]
            candidate_features = [
                c
                for c in candidate_features
                if c not in meta_df.columns or not meta_df[c].isna().all()
            ]
            meta_model, meta_feature_cols, meta_metrics = await asyncio.to_thread(
                train_meta_model,
                meta_df,
                candidate_features,
                settings.META_LABELING_MIN_TRADES,
            )
            if meta_model is not None:
                logger.info(f"[Meta-Labeling] Модель прошла собственный гейт: {meta_metrics}")
            else:
                logger.info(f"[Meta-Labeling] Модель отклонена: {meta_metrics.get('rejected_reason')}")
        except Exception as meta_err:
            logger.error(f"[Meta-Labeling] Ошибка обучения вторичной модели: {meta_err}")

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
        "holdout_f1": float(holdout_f1) if holdout_f1 is not None else None,
        "economic_test_f1": float(holdout_f1) if holdout_f1 is not None else None,
        "total_folds": fold_count,
        "total_test_samples": len(all_y_true),
        "economic_backtest": economic_backtest_metrics,
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
        "wfo_protocol": "nested_inner_optuna_calibration_economic_test",
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
        "edge_threshold": edge_threshold,
        "edge_threshold_sweep": edge_sweep,
        "backtest_metrics": economic_backtest_metrics,
        "evaluation_protocol": {
            "name": "nested_wfo",
            "optuna_data": "inner_train_only",
            "calibration_data": "calibration_only",
            "economic_metrics_data": "economic_test_only",
        },
        "meta_model": meta_model,
        "meta_features": meta_feature_cols,
        "meta_metrics": meta_metrics,
        "regime_drift_pvalue_at_train": regime_drift_pvalue,
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
