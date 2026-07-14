import os
import json
import pickle
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models.backtest import TimeSeriesWalkForwardSplitter
from src.crud.experiment import ExperimentRepository
from src.datasets.build import get_git_sha
from src.core.config import get_settings

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
        }

        splitter = TimeSeriesWalkForwardSplitter(
            train_size=train_size, test_size=test_size
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
    study.optimize(objective, n_trials=n_trials)

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
    """
    # 1. Загружаем датасет и его описание
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

    # 1.5. Шаг автоматической калибровки параметров через Optuna (если включено в конфиге)
    best_params = {}
    if settings.OPTUNA_TUNING_ENABLED:
        logger.info(
            f"[*] Запуск подбора гиперпараметров через Optuna ({settings.OPTUNA_TRIALS} попыток)..."
        )
        best_params = await tune_lgbm_hyperparameters(
            session=session,
            df_clean=df_clean,
            feature_cols=feature_cols,
            target_col=target_col,
            train_size=train_size,
            test_size=test_size,
            metadata_version=metadata["version"],
            n_trials=settings.OPTUNA_TRIALS,
        )
        logger.info(f"[+] Лучшие параметры подобраны: {best_params}")

    splitter = TimeSeriesWalkForwardSplitter(train_size=train_size, test_size=test_size)

    all_y_true = []
    all_y_pred = []
    fold_count = 0

    final_model = None
    oos_dfs = []  # Список для сбора чистых Out-of-Sample данных

    is_multiclass = target_col == "target_triple"
    avg_method = "macro" if is_multiclass else "binary"

    # 2. Walk-Forward цикл обучения
    for train_df, test_df, info in splitter.split(df_clean):
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        # Для тройной классификации переводим метки в [0, 1, 2]
        if is_multiclass:
            y_train = y_train.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
            y_test = y_test.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
        else:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

        model_kwargs = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
            "verbosity": -1,
        }
        if settings.OPTUNA_TUNING_ENABLED and best_params:
            model_kwargs.update(best_params)

        model = LGBMClassifier(**model_kwargs)
        model.fit(X_train, y_train)

        # Делаем предсказание OOS
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        # Собираем OOS-предсказания с ценами без утечки данных
        test_df_copy = test_df.copy()
        if is_multiclass:
            # Декодируем предсказанные классы обратно в торговые сигналы
            signal_map = {0: -1.0, 1: 0.0, 2: 1.0}
            test_df_copy["predicted_signal"] = pd.Series(
                y_pred, index=test_df.index
            ).map(signal_map)
        else:
            test_df_copy["predicted_signal"] = y_pred

        oos_dfs.append(test_df_copy)

        fold_count += 1
        final_model = model

    if fold_count == 0:
        raise ValueError("Не удалось запустить Walk-Forward. Проверьте размер данных.")

    # Объединяем все OOS-фолды в один чистый датасет для калибровки
    df_oos = pd.concat(oos_dfs).sort_values("open_time").reset_index(drop=True)

    # 3. Рассчитываем метрики точности
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

    # 4. Сохраняем результаты в БД экспериментов
    repo = ExperimentRepository(session)
    experiment = await repo.log_experiment(
        model_name="LightGBM_Model",
        dataset_version=metadata["version"],
        parameters=parameters,
        metrics=metrics,
        git_sha=get_git_sha(),
    )

    # 5. Сохраняем файл модели и OOS-данные на диск
    os.makedirs(models_dir, exist_ok=True)
    clean_symbol = metadata["symbol"].replace("/", "").replace(":", "")
    model_filename = f"lgbm_{clean_symbol}_{metadata['timeframe'].replace('/', '')}.pkl"
    model_path = os.path.join(models_dir, model_filename)

    saved_data = {
        "model": final_model,
        "features": feature_cols,
        "scaler": None,
        "symbol": metadata["symbol"],
        "timeframe": metadata["timeframe"],
        "version": metadata["version"],
        "target_col": target_col,
        "df_oos": df_oos,  # ← СОХРАНЯЕМ ЧИСТЫЕ OOS ДАННЫЕ В МОДЕЛЬ
    }

    with open(model_path, "wb") as f:
        pickle.dump(saved_data, f)

    return {
        "experiment_id": experiment.id,
        "model_path": model_path,
        "parameters": parameters,
        "metrics": metrics,
    }