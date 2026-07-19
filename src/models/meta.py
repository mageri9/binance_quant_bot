import pandas as pd
from lightgbm import LGBMClassifier

from src.strategy.signals import simulate_strategy
from sklearn.metrics import precision_score, recall_score


META_BASE_FEATURES = ["adx", "atr_pct", "volume_ratio", "volatility"]


def build_meta_dataset(
    df_oos: pd.DataFrame,
    predicted_col: str = "predicted_signal",
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Строит обучающий датасет для вторичной модели из OOS-предсказаний
    первичной. Каждая строка — одна фактически совершённая сделка
    первичной модели; success=1, если сделка прибыльна в честной
    симуляции simulate_strategy, иначе 0.
    """
    required = set(META_BASE_FEATURES + [predicted_col, "close", "high", "low"])
    missing = required - set(df_oos.columns)
    if missing:
        raise ValueError(f"df_oos не содержит колонки для meta-labeling: {missing}")

    df_reset = df_oos.reset_index(drop=True)
    _, trades_df = simulate_strategy(
        df_reset, predicted_col=predicted_col,
        transaction_cost=transaction_cost, return_trade_log=True,
    )

    cols = META_BASE_FEATURES + [predicted_col, "predicted_confidence", "success"]
    if trades_df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, trade in trades_df.iterrows():
        entry_idx = int(trade["entry_idx"])
        entry_row = df_reset.iloc[entry_idx]
        row = {feat: entry_row[feat] for feat in META_BASE_FEATURES}
        row[predicted_col] = entry_row[predicted_col]
        row["predicted_confidence"] = entry_row.get("predicted_confidence", None)
        row["success"] = int(trade["return"] > 0)
        rows.append(row)

    return pd.DataFrame(rows)


def train_meta_model(
    meta_df: pd.DataFrame,
    feature_cols: list[str],
    min_trades: int = 30,
    holdout_frac: float = 0.3,
) -> tuple[LGBMClassifier | None, list[str] | None, dict]:
    """
    Обучает бинарный классификатор "успешна ли сделка первичной модели"
    с честной проверкой качества на собственном holdout-срезе (хронологическом,
    т.к. meta_df уже упорядочен по времени входа в сделку).

    Meta-модель встраивается в прод, только если на holdout она реально
    повышает долю успешных сделок среди тех, что сама одобряет (proba>=0.5),
    по сравнению с базовой долей успеха без фильтра. Иначе — не тренируем зря
    рискованный фильтр, который может резать хорошие сигналы наугад.
    """
    metrics = {"n_trades": len(meta_df)}

    if len(meta_df) < min_trades:
        metrics["rejected_reason"] = "insufficient_trades"
        return None, None, metrics

    meta_df = meta_df.reset_index(drop=True)
    split_idx = int(len(meta_df) * (1 - holdout_frac))
    train_df = meta_df.iloc[:split_idx]
    holdout_df = meta_df.iloc[split_idx:]

    if len(train_df) < 5 or len(holdout_df) < 5:
        metrics["rejected_reason"] = "insufficient_holdout_split"
        return None, None, metrics

    if train_df["success"].nunique() < 2 or holdout_df["success"].nunique() < 2:
        metrics["rejected_reason"] = "single_class_in_split"
        return None, None, metrics

    def _new_model():
        return LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=-1, n_jobs=1, class_weight="balanced",
        )

    probe_model = _new_model()
    probe_model.fit(train_df[feature_cols], train_df["success"])

    holdout_pred = probe_model.predict(holdout_df[feature_cols])
    holdout_proba = probe_model.predict_proba(holdout_df[feature_cols])
    classes = list(probe_model.classes_)
    success_idx = classes.index(1) if 1 in classes else 1
    holdout_success_proba = holdout_proba[:, success_idx]

    baseline_success_rate = float(holdout_df["success"].mean())
    approved_mask = holdout_success_proba >= 0.5
    n_approved = int(approved_mask.sum())
    approved_success_rate = (
        float(holdout_df.loc[approved_mask, "success"].mean()) if n_approved > 0 else None
    )

    metrics.update({
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "precision": float(precision_score(holdout_df["success"], holdout_pred, zero_division=0)),
        "recall": float(recall_score(holdout_df["success"], holdout_pred, zero_division=0)),
        "baseline_success_rate": baseline_success_rate,
        "approved_success_rate": approved_success_rate,
        "n_approved": n_approved,
    })

    if approved_success_rate is None or approved_success_rate <= baseline_success_rate:
        metrics["rejected_reason"] = "no_lift_over_baseline"
        return None, None, metrics

    # Прошла собственный гейт — дообучаем на всех данных для продакшна
    final_model = _new_model()
    final_model.fit(meta_df[feature_cols], meta_df["success"])
    metrics["rejected_reason"] = None
    return final_model, feature_cols, metrics