import os
import json
import asyncio
import pickle
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models.backtest import TimeSeriesWalkForwardSplitter, purge_train_tail
from src.models.meta import (
    PRIMARY_OOF_FOLD_COLUMN,
    PRIMARY_OOF_ROW_COLUMN,
    PRIMARY_TRAIN_END_COLUMN,
)
from src.models.regimes import fit_soft_regime_ensemble, soft_regime_weights
from src.crud.experiment import ExperimentRepository
from src.datasets.build import get_git_sha
from src.core.config import get_settings
from src.labels.generator import MAX_ADAPTIVE_HORIZON_CANDLES
from datetime import datetime, timezone
from src.strategy.signals import simulate_strategy
from src.execution.trade import trade_spec_from_settings
from src.models.economic import EconomicReturnRegressor, ECONOMIC_TARGETS
from src.models.economic_quality import economic_quality_failure
from src.strategy.edge import apply_edge_threshold, sweep_edge_thresholds
from src.utils.artifact_paths import get_oos_path
from src.models.artifacts import MODEL_ARTIFACT_SCHEMA_VERSION, feature_hash


async def _run_economic_return_experiment(
    session, df_clean, metadata, feature_cols, models_dir, learning_rate,
    n_estimators, max_depth, settings, bypass_quality_gates,
) -> dict:
    """Train two regressors and select a side only from its expected net PnL."""
    if any(column not in df_clean for column in ECONOMIC_TARGETS):
        raise ValueError("Economic target requires long_net_return and short_net_return columns")
    if len(df_clean) < 3:
        raise ValueError("Economic return training requires at least three resolved trades")

    # Calibrate the EV threshold before evaluating on the final chronological split.
    calibration_start = max(1, int(len(df_clean) * 0.6))
    test_start = min(max(calibration_start + 1, int(len(df_clean) * 0.8)), len(df_clean) - 1)
    train_df = df_clean.iloc[:calibration_start]
    calibration_df = df_clean.iloc[calibration_start:test_start]
    test_df = df_clean.iloc[test_start:]
    model_kwargs = dict(
        learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, verbosity=-1, n_jobs=1,
    )
    selection_model = EconomicReturnRegressor(**model_kwargs)
    await asyncio.to_thread(selection_model.fit, train_df[feature_cols], train_df[list(ECONOMIC_TARGETS)])

    def prediction_frame(source_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        result = source_df.copy()
        long_pred, short_pred = selection_model.predict_returns(result[feature_cols])
        result["predicted_long_return"] = long_pred
        result["predicted_short_return"] = short_pred
        result["predicted_expected_return"] = np.maximum(long_pred, short_pred)
        result["predicted_signal"] = np.where(
            long_pred > short_pred, 1, np.where(short_pred > long_pred, -1, 0)
        )
        result = apply_edge_threshold(result, 0.0)
        result["oos_split"] = split_name
        return result

    calibration_oos = prediction_frame(calibration_df, "calibration")
    test_oos = prediction_frame(test_df, "economic_test")
    simulate_kwargs = {
        "trade_spec": trade_spec_from_settings(settings),
        "stop_risk_pct": settings.BACKTEST_STOP_RISK_PCT,
        "target_volatility": settings.BACKTEST_TARGET_VOLATILITY,
        "max_position_pct": settings.BACKTEST_MAX_POSITION_PCT,
    }
    minimum_ev = max(0.0, float(settings.MIN_EXPECTED_RETURN))
    edge_sweep: list[dict] = []
    if settings.EDGE_THRESHOLD_SWEEP_ENABLED and not calibration_oos.empty:
        minimum_ev, edge_sweep = await asyncio.to_thread(
            sweep_edge_thresholds, calibration_oos, settings.EDGE_THRESHOLD_GRID,
            settings.EDGE_MIN_COVERAGE, settings.CALIBRATION_MIN_TRADES, simulate_kwargs,
        )
    test_oos = apply_edge_threshold(test_oos, minimum_ev)
    economic_backtest = simulate_strategy(test_oos, **simulate_kwargs)
    if not bypass_quality_gates:
        rejection = economic_quality_failure(
            economic_backtest, min_trades=settings.ECONOMIC_GATE_MIN_TRADES,
        )
        if rejection:
            raise ValueError(f"LGBM model REJECTED by Economic Quality Gate: {rejection}")
    actual_side = np.where(
        test_oos["long_net_return"].to_numpy() >= test_oos["short_net_return"].to_numpy(), 1, -1
    )
    directional_accuracy = float(np.mean(test_oos["predicted_signal"].to_numpy() == actual_side))
    metrics = {
        "mae": float(mean_absolute_error(test_oos["expected_return"], test_oos["predicted_expected_return"])),
        "directional_accuracy": directional_accuracy,
        "accuracy": directional_accuracy,
        "precision": directional_accuracy,
        "recall": directional_accuracy,
        # Retained as a compatibility metric for the promotion pipeline.
        "f1": directional_accuracy,
        "holdout_f1": directional_accuracy,
        "economic_test_f1": directional_accuracy,
        "total_folds": 1,
        "total_test_samples": len(test_df),
        "economic_backtest": economic_backtest,
    }
    final_train_df = pd.concat([train_df, calibration_df], ignore_index=True)
    model = EconomicReturnRegressor(**model_kwargs)
    await asyncio.to_thread(model.fit, final_train_df[feature_cols], final_train_df[list(ECONOMIC_TARGETS)])
    os.makedirs(models_dir, exist_ok=True)
    clean_symbol = metadata["symbol"].replace("/", "").replace(":", "")
    clean_tf = metadata["timeframe"].replace("/", "")
    model_path = os.path.join(models_dir, f"lgbm_{clean_symbol}_{clean_tf}.pkl")
    artifact = {
        "model_id": f"economic_return_{clean_symbol}_{clean_tf}_{metadata['version']}",
        "dataset_version": metadata["version"],
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "model_type": "economic_return_regression",
        "model": model,
        "target_col": "expected_return",
        "features": feature_cols,
        "features_hash": feature_hash(feature_cols),
        "min_expected_return": minimum_ev,
        "edge_threshold_sweep": edge_sweep,
        "calibration": {"sl_pct": settings.TRADE_SL_PCT, "tp_pct": settings.TRADE_TP_PCT,
                        "horizon": settings.TRADE_TIMEOUT_CANDLES},
        "backtest_metrics": economic_backtest,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    await asyncio.to_thread(test_oos.to_parquet, get_oos_path(model_path), index=False)
    experiment = await ExperimentRepository(session).log_experiment(
        model_name="LightGBM_EconomicReturnRegressor", dataset_version=metadata["version"],
        parameters={"model_type": "economic_return_regression", "features_used": feature_cols,
                    "min_expected_return": minimum_ev},
        metrics=metrics, git_sha=get_git_sha(),
    )
    return {"experiment_id": experiment.id, "model_path": model_path,
            "parameters": {"model_type": "economic_return_regression"}, "metrics": metrics}

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
    objective_metric: str = "sharpe",
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

    if objective_metric not in ("sharpe", "sortino", "expectancy", "profit_factor", "utility"):
        raise ValueError(f"Неизвестный objective_metric: {objective_metric}")

    def _fold_score(y_test, y_pred, test_df):
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
            trade_spec=trade_spec_from_settings(settings),
            stop_risk_pct=settings.BACKTEST_STOP_RISK_PCT,
            target_volatility=settings.BACKTEST_TARGET_VOLATILITY,
            max_position_pct=settings.BACKTEST_MAX_POSITION_PCT,
        )
        if sim_metrics["total_trades"] < settings.OPTUNA_MIN_TRADES:
            return -1e6
        scores = {
            "sharpe": sim_metrics["sharpe_ci_low"],
            "sortino": sim_metrics["sortino_ratio"],
            "expectancy": sim_metrics["expectancy"],
            "profit_factor": sim_metrics["profit_factor"],
            "utility": sim_metrics["expectancy"] + sim_metrics["sharpe_ci_low"] - sim_metrics["max_drawdown"],
        }
        return scores[objective_metric]

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
    label_horizon, model_kwargs, is_multiclass, avg_method, use_soft_regimes,
    regime_temperature,
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

        if use_soft_regimes:
            model = fit_soft_regime_ensemble(
                X_train, y_train, model_kwargs, regime_temperature,
            )
        else:
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
        if use_soft_regimes:
            # Persist the OOF blend used for this row for regime diagnostics.
            memberships = soft_regime_weights(X_test, regime_temperature)
            for regime, values in memberships.items():
                test_df_copy[f"regime_weight_{regime}"] = values.to_numpy()
        # Preserve proof that each primary prediction was generated out-of-fold.
        test_df_copy[PRIMARY_OOF_FOLD_COLUMN] = info["fold"]
        test_df_copy[PRIMARY_TRAIN_END_COLUMN] = info["train_end_idx"] - label_horizon - 1
        test_df_copy[PRIMARY_OOF_ROW_COLUMN] = test_df.index

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

    if target_col != "expected_return":
        raise ValueError(
            "Classification targets cannot authorize trades. Set TARGET_COL=expected_return "
            "and rebuild the dataset with post-cost economic targets."
        )
    required_economic_targets = {"expected_return", *ECONOMIC_TARGETS}
    if missing := required_economic_targets.difference(df.columns):
        raise ValueError(f"Dataset is missing economic targets: {sorted(missing)}")

    feature_cols = metadata["features"]

    df_clean = df.dropna(subset=feature_cols + list(ECONOMIC_TARGETS)).reset_index(drop=True)

    if target_col == "expected_return":
        return await _run_economic_return_experiment(
            session, df_clean, metadata, feature_cols, models_dir, learning_rate,
            n_estimators, max_depth, settings, bypass_quality_gates,
        )

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
            objective_metric=settings.OPTUNA_OBJECTIVE_METRIC,
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

    use_soft_regimes = bool(getattr(settings, "SOFT_REGIME_ENSEMBLE_ENABLED", True))
    regime_temperature = float(getattr(settings, "SOFT_REGIME_TEMPERATURE", 1.0))

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
        use_soft_regimes,
        regime_temperature,
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
        X_train_val = df_train_val[feature_cols]
        y_train_val = df_train_val[target_col]
        if is_multiclass:
            y_train_val_mapped = y_train_val.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
        else:
            y_train_val_mapped = y_train_val.astype(int)
        if use_soft_regimes:
            gate_candidate = await asyncio.to_thread(
                fit_soft_regime_ensemble, X_train_val, y_train_val_mapped,
                model_kwargs, regime_temperature,
            )
        else:
            gate_candidate = LGBMClassifier(**model_kwargs)
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
        # Quality Gate 2: Защита от вырождения предсказаний (Class Collapse Protection)
        unique_preds, counts = np.unique(y_holdout_pred, return_counts=True)
        pred_ratios = counts / len(y_holdout_pred)
        logger.info(
            f"[MLOps Train] Модель успешно прошла все Quality Gates на Holdout. Holdout F1: {holdout_f1:.4f}"
        )

    # --- 4. ФИНАЛЬНОЕ ДООБУЧЕНИЕ НА 100% ДАННЫХ (FIT ON ALL) ---
    logger.info(
        "[MLOps Train] Запуск финального дообучения модели на 100% исторических данных..."
    )
    # Produce downstream predictions from an inner-train-only candidate. The
    # economic segment is never used while selecting risk parameters.
    y_inner = df_inner_train[target_col]
    if is_multiclass:
        y_inner = y_inner.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    else:
        y_inner = y_inner.astype(int)
    if use_soft_regimes:
        partition_candidate = await asyncio.to_thread(
            fit_soft_regime_ensemble, df_inner_train[feature_cols], y_inner,
            model_kwargs, regime_temperature,
        )
    else:
        partition_candidate = LGBMClassifier(**model_kwargs)
        await asyncio.to_thread(
            partition_candidate.fit, df_inner_train[feature_cols], y_inner,
        )

    def _prediction_frame(source_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        result = source_df.copy()
        predicted = partition_candidate.predict(result[feature_cols])
        probabilities = partition_candidate.predict_proba(result[feature_cols])
        if is_multiclass:
            result["predicted_signal"] = pd.Series(predicted, index=result.index).map(SIGNAL_MAP_TRIPLE)
        else:
            result["predicted_signal"] = predicted
        result["predicted_confidence"] = probabilities.max(axis=1)
        if use_soft_regimes:
            memberships = soft_regime_weights(result[feature_cols], regime_temperature)
            for regime, values in memberships.items():
                result[f"regime_weight_{regime}"] = values.to_numpy()
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
                "trade_spec": trade_spec_from_settings(settings),
                "stop_risk_pct": settings.BACKTEST_STOP_RISK_PCT,
                "target_volatility": settings.BACKTEST_TARGET_VOLATILITY,
                "max_position_pct": settings.BACKTEST_MAX_POSITION_PCT,
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
        trade_spec=trade_spec_from_settings(settings),
        stop_risk_pct=settings.BACKTEST_STOP_RISK_PCT,
        target_volatility=settings.BACKTEST_TARGET_VOLATILITY,
        max_position_pct=settings.BACKTEST_MAX_POSITION_PCT,
    )
    if not bypass_quality_gates:
        rejection = economic_quality_failure(
            economic_backtest_metrics,
            min_trades=settings.ECONOMIC_GATE_MIN_TRADES,
        )
        if rejection:
            raise ValueError(f"LGBM model REJECTED by Economic Quality Gate: {rejection}")

    df_final_train = pd.concat([df_inner_train, df_calibration], ignore_index=True)
    X_all = df_final_train[feature_cols]
    y_all = df_final_train[target_col]
    if is_multiclass:
        y_all_mapped = y_all.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    else:
        y_all_mapped = y_all.astype(int)

    if use_soft_regimes:
        final_model = await asyncio.to_thread(
            fit_soft_regime_ensemble, X_all, y_all_mapped, model_kwargs,
            regime_temperature,
        )
    else:
        final_model = LGBMClassifier(**model_kwargs)
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
                build_cross_fitted_meta_dataset,
                train_meta_model,
                META_BASE_FEATURES,
            )

            meta_df = await asyncio.to_thread(
                build_cross_fitted_meta_dataset,
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
                meta_threshold=settings.META_LABELING_THRESHOLD,
            )
            meta_metrics["cross_fitted_primary_oof"] = True
            meta_metrics["primary_oof_folds"] = int(
                inner_oos[PRIMARY_OOF_FOLD_COLUMN].nunique()
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
        "model_type": "economic_return_regression",
        "base_model_type": "LightGBM",
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
        "soft_regime_ensemble": use_soft_regimes,
        "soft_regime_temperature": regime_temperature if use_soft_regimes else None,
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

    features_hash = feature_hash(feature_cols)

    artifact = {
        "model_id": f"lgbm_{clean_symbol}_{clean_tf}_{metadata['version']}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": metadata["symbol"],
        "timeframe": metadata["timeframe"],
        "dataset_version": metadata["version"],
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "git_sha": get_git_sha(),
        "target_col": target_col,
        "features": feature_cols,
        "features_hash": features_hash,
        "model_type": "economic_return_regression",
        "model": final_model,
        "soft_regime_ensemble": {
            "enabled": use_soft_regimes,
            "regimes": ["bear", "range", "bull"] if use_soft_regimes else [],
            "temperature": regime_temperature if use_soft_regimes else None,
        },
        "scaler": None,
        "calibration": {
            "sl_pct": settings.TRADE_SL_PCT,
            "tp_pct": settings.TRADE_TP_PCT,
            "horizon": settings.TRADE_TIMEOUT_CANDLES,
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
