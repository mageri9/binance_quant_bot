import pandas as pd
from lightgbm import LGBMClassifier

from src.strategy.signals import simulate_strategy

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
    meta_df: pd.DataFrame, feature_cols: list[str], min_trades: int = 30
) -> tuple[LGBMClassifier | None, list[str] | None]:
    """
    Обучает бинарный классификатор "успешна ли сделка первичной модели".
    Возвращает (None, None), если сделок недостаточно или в выборке
    только один класс (нельзя обучить бинарный классификатор).
    """
    if len(meta_df) < min_trades:
        return None, None

    y = meta_df["success"].astype(int)
    if y.nunique() < 2:
        return None, None

    X = meta_df[feature_cols]
    model = LGBMClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=-1, n_jobs=1, class_weight="balanced",
    )
    model.fit(X, y)
    return model, feature_cols