import pandas as pd
from lightgbm import LGBMRegressor

from src.strategy.signals import simulate_strategy
from sklearn.metrics import mean_absolute_error


META_BASE_FEATURES = ["adx", "atr_pct", "volume_ratio", "volatility"]
PRIMARY_OOF_FOLD_COLUMN = "primary_oof_fold"
PRIMARY_TRAIN_END_COLUMN = "primary_train_end_idx"
PRIMARY_OOF_ROW_COLUMN = "primary_oof_row_idx"


def build_meta_dataset(
    df_oos: pd.DataFrame,
    predicted_col: str = "predicted_signal",
    transaction_cost: float = 0.001,
    drift_pvalue: float | None = None,
) -> pd.DataFrame:
    """
    Строит обучающий датасет для вторичной модели из OOS-предсказаний
    первичной. Каждая строка — одна фактически совершённая сделка
    первичной модели. Цель ``future_net_return`` - фактическая post-cost
    доходность сделки из честной симуляции, а не бинарный признак выигрыша.
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

    if trades_df.empty:
        cols = META_BASE_FEATURES + [predicted_col, "predicted_confidence", "future_net_return"]
        if drift_pvalue is not None:
            cols.append("regime_drift_pvalue")
        return pd.DataFrame(columns=cols)

    rows = []
    for _, trade in trades_df.iterrows():
        entry_idx = int(trade["entry_idx"])
        entry_row = df_reset.iloc[entry_idx]
        row = {feat: entry_row[feat] for feat in META_BASE_FEATURES}
        row[predicted_col] = entry_row[predicted_col]
        row["predicted_confidence"] = entry_row.get("predicted_confidence", None)
        row["future_net_return"] = float(trade["return"])
        if drift_pvalue is not None:
            row["regime_drift_pvalue"] = drift_pvalue
        rows.append(row)

    return pd.DataFrame(rows)


def build_cross_fitted_meta_dataset(
    primary_oof: pd.DataFrame,
    predicted_col: str = "predicted_signal",
    transaction_cost: float = 0.001,
    drift_pvalue: float | None = None,
) -> pd.DataFrame:
    """Build meta labels exclusively from primary-model out-of-fold predictions.

    Meta-labeling is a trade selector, not a new alpha source.  Each candidate
    signal must therefore have been emitted by a primary model trained strictly
    before its OOS row.  The walk-forward trainer writes this provenance into
    the frame; rejecting an unprovenanced frame makes accidental in-sample
    primary predictions impossible to feed into the secondary model.
    """
    required = {
        PRIMARY_OOF_FOLD_COLUMN,
        PRIMARY_TRAIN_END_COLUMN,
        PRIMARY_OOF_ROW_COLUMN,
    }
    missing = required - set(primary_oof.columns)
    if missing:
        raise ValueError(f"primary_oof lacks cross-fitted provenance: {missing}")

    if primary_oof.empty:
        return build_meta_dataset(
            primary_oof, predicted_col=predicted_col,
            transaction_cost=transaction_cost, drift_pvalue=drift_pvalue,
        )

    row_positions = pd.to_numeric(primary_oof[PRIMARY_OOF_ROW_COLUMN], errors="coerce")
    train_end = pd.to_numeric(primary_oof[PRIMARY_TRAIN_END_COLUMN], errors="coerce")
    invalid_provenance = (
        row_positions.isna().any()
        or train_end.isna().any()
        or (train_end >= row_positions).any()
    )
    if invalid_provenance:
        raise ValueError("primary_oof contains a non-cross-fitted primary prediction")

    return build_meta_dataset(
        primary_oof, predicted_col=predicted_col,
        transaction_cost=transaction_cost, drift_pvalue=drift_pvalue,
    )


def train_meta_model(
    meta_df: pd.DataFrame,
    feature_cols: list[str],
    min_trades: int = 30,
    holdout_frac: float = 0.3,
    meta_threshold: float = 0.0,
) -> tuple[LGBMRegressor | None, list[str] | None, dict]:
    """
    Обучает регрессию будущего post-cost net return сделки первичной модели.
    Модель допускается в прод только если её gate повышает фактический
    expectancy одобренных сделок на хронологическом holdout.
    """
    metrics = {"n_trades": len(meta_df)}

    if "future_net_return" not in meta_df:
        raise ValueError("meta_df must contain future_net_return")

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

    def _new_model():
        return LGBMRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, verbosity=-1, n_jobs=1,
        )

    probe_model = _new_model()
    probe_model.fit(train_df[feature_cols], train_df["future_net_return"])

    holdout_pred = probe_model.predict(holdout_df[feature_cols])
    baseline_expectancy = float(holdout_df["future_net_return"].mean())
    approved_mask = holdout_pred >= meta_threshold
    n_approved = int(approved_mask.sum())
    approved_expectancy = (
        float(holdout_df.loc[approved_mask, "future_net_return"].mean()) if n_approved > 0 else None
    )

    metrics.update({
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "holdout_mae": float(mean_absolute_error(holdout_df["future_net_return"], holdout_pred)),
        "baseline_expectancy": baseline_expectancy,
        "approved_expectancy": approved_expectancy,
        "meta_threshold": meta_threshold,
        "n_approved": n_approved,
    })

    if approved_expectancy is None or approved_expectancy <= baseline_expectancy:
        metrics["rejected_reason"] = "no_expectancy_lift_over_baseline"
        return None, None, metrics

    # Прошла собственный гейт — дообучаем на всех данных для продакшна
    final_model = _new_model()
    final_model.fit(meta_df[feature_cols], meta_df["future_net_return"])
    metrics["rejected_reason"] = None
    return final_model, feature_cols, metrics

def apply_meta_gate(
    df: pd.DataFrame,
    meta_model,
    meta_features: list[str] | None,
    meta_threshold: float = 0.0,
    predicted_col: str = "predicted_signal",
) -> pd.Series:
    """
    Возвращает копию колонки сигналов, где сигналы с недостаточным
    предсказанным future net return погашены до 0 (HOLD).
    Если meta_model/meta_features отсутствуют — возвращает сигналы без изменений.
    """
    gated = df[predicted_col].copy()
    if meta_model is None or not meta_features:
        return gated

    mask_nonzero = gated != 0
    if not mask_nonzero.any():
        return gated

    sub = df.loc[mask_nonzero]
    missing = [f for f in meta_features if f not in sub.columns and f != predicted_col]
    if missing:
        return gated  # недостаточно фичей для честного гейтинга — не гейтим вслепую

    X = sub[[f for f in meta_features if f != predicted_col]].copy()
    if predicted_col in meta_features:
        X[predicted_col] = sub[predicted_col]
    X = X[meta_features]

    expected_returns = meta_model.predict(X)
    reject_idx = sub.index[expected_returns < meta_threshold]
    gated.loc[reject_idx] = 0
    return gated
