"""Walk-forward comparison of directional classifiers and return regressors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.execution.kernel import ExecutionKernel
from src.execution.trade import TradeSpec, evaluate_trade
from src.models.backtest import TimeSeriesWalkForwardSplitter
from src.strategy.signals import simulate_strategy


@dataclass(frozen=True)
class SideComparisonConfig:
    train_size: int
    test_size: int
    horizon: int = 5
    classifier_threshold: float = 0.55
    quantile: float = 0.25
    label_horizon: int = 15
    trade_spec: TradeSpec | None = None


def build_side_net_returns(
    df: pd.DataFrame,
    *,
    horizon: int,
    execution_kernel: ExecutionKernel,
    trade_spec: TradeSpec | None = None,
) -> pd.DataFrame:
    """Label each candle with the net long and short return over a fixed horizon.

    Signals are observed at candle *t*, entered at the next open, and exited at
    the open after ``horizon`` completed bars.  This matches the signal timing
    used by ``simulate_strategy`` while keeping the regression target independent
    of the model's own entry decisions.
    """
    required = {"open"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for net-return labels: {sorted(missing)}")

    result = pd.DataFrame(index=df.index, columns=["long_net_return", "short_net_return"], dtype=float)
    # Legacy arguments are adapted at this edge; target economics themselves
    # are resolved exclusively by the same TradeSpec evaluator as training.
    if trade_spec is None and not {"high", "low"}.issubset(df.columns):
        # Historical comparison datasets contain opens only.  Keep that adapter
        # outside the canonical path while making the missing intrabar range
        # deterministic (no barrier can be hit).
        opens = df["open"].to_numpy(dtype=float)
        for signal_idx in range(len(df) - horizon - 2):
            entry = opens[signal_idx + 1]
            exit_price = opens[signal_idx + horizon + 2]
            long_entry = execution_kernel.market_fill(side="buy", reference_price=entry, amount=1)
            long_exit = execution_kernel.market_fill(side="sell", reference_price=exit_price, amount=1)
            short_entry = execution_kernel.market_fill(side="sell", reference_price=entry, amount=1)
            short_exit = execution_kernel.market_fill(side="buy", reference_price=exit_price, amount=1)
            result.iloc[signal_idx, 0] = float(execution_kernel.realized_return(
                entry=long_entry, exit=long_exit, is_short=False,
            ))
            result.iloc[signal_idx, 1] = float(execution_kernel.realized_return(
                entry=short_entry, exit=short_exit, is_short=True,
            ))
        return result
    else:
        spec = trade_spec or TradeSpec(timeout=horizon, costs=execution_kernel.costs)
    for signal_idx in range(len(df)):
        long = evaluate_trade(df, signal_idx, "LONG", spec)
        short = evaluate_trade(df, signal_idx, "SHORT", spec)
        if long is not None:
            result.iloc[signal_idx, 0] = long.net_return
        if short is not None:
            result.iloc[signal_idx, 1] = short.net_return
    return result


def _signals_from_classifier(model_long, model_short, X: pd.DataFrame, threshold: float) -> np.ndarray:
    def positive_probability(model) -> np.ndarray:
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        # A monotonic period can make one side entirely profitable/unprofitable.
        # LightGBM then emits a one-column probability array.
        return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

    long_prob = positive_probability(model_long)
    short_prob = positive_probability(model_short)
    best = np.maximum(long_prob, short_prob)
    return np.where(best < threshold, 0, np.where(long_prob >= short_prob, 1, -1))


def _signals_from_returns(long_values: np.ndarray, short_values: np.ndarray) -> np.ndarray:
    best = np.maximum(long_values, short_values)
    return np.where(best <= 0.0, 0, np.where(long_values >= short_values, 1, -1))


def _model_metrics(test_df: pd.DataFrame, signals: np.ndarray, trade_spec: TradeSpec) -> dict:
    simulation = test_df.copy()
    simulation["predicted_signal"] = signals
    return simulate_strategy(
        simulation,
        predicted_col="predicted_signal",
        trade_spec=trade_spec,
    )


def run_side_model_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: SideComparisonConfig,
    execution_kernel: ExecutionKernel,
) -> dict:
    """Compare side classifiers, conditional mean, and conditional quantile models.

    All policies see the same folds and select at most one side per candle.  The
    returned deltas are relative to the side-specific classification policy.
    """
    trade_spec = config.trade_spec or TradeSpec(
        timeout=config.horizon, costs=execution_kernel.costs,
    )
    labels = build_side_net_returns(
        df, horizon=config.horizon, execution_kernel=execution_kernel, trade_spec=trade_spec,
    )
    prepared = pd.concat([df.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    prepared = prepared.dropna(subset=feature_cols + ["long_net_return", "short_net_return"]).reset_index(drop=True)
    splitter = TimeSeriesWalkForwardSplitter(
        train_size=config.train_size,
        test_size=config.test_size,
        label_horizon=config.label_horizon,
    )
    per_policy: dict[str, list[dict]] = {"side_classifier": [], "net_return_regression": [], "quantile_regression": []}
    fold_count = 0

    for train_df, test_df, _ in splitter.split(prepared):
        X_train, X_test = train_df[feature_cols], test_df[feature_cols]
        classifier_kwargs = {"n_estimators": 100, "learning_rate": 0.05, "random_state": 42, "verbosity": -1, "n_jobs": 1, "class_weight": "balanced"}
        regressor_kwargs = {"n_estimators": 100, "learning_rate": 0.05, "random_state": 42, "verbosity": -1, "n_jobs": 1}

        long_cls = LGBMClassifier(**classifier_kwargs).fit(X_train, (train_df["long_net_return"] > 0).astype(int))
        short_cls = LGBMClassifier(**classifier_kwargs).fit(X_train, (train_df["short_net_return"] > 0).astype(int))
        long_mean = LGBMRegressor(**regressor_kwargs).fit(X_train, train_df["long_net_return"])
        short_mean = LGBMRegressor(**regressor_kwargs).fit(X_train, train_df["short_net_return"])
        quantile_kwargs = {**regressor_kwargs, "objective": "quantile", "alpha": config.quantile}
        long_quantile = LGBMRegressor(**quantile_kwargs).fit(X_train, train_df["long_net_return"])
        short_quantile = LGBMRegressor(**quantile_kwargs).fit(X_train, train_df["short_net_return"])

        signals = {
            "side_classifier": _signals_from_classifier(long_cls, short_cls, X_test, config.classifier_threshold),
            "net_return_regression": _signals_from_returns(long_mean.predict(X_test), short_mean.predict(X_test)),
            "quantile_regression": _signals_from_returns(long_quantile.predict(X_test), short_quantile.predict(X_test)),
        }
        for policy, policy_signals in signals.items():
            per_policy[policy].append(_model_metrics(test_df, policy_signals, trade_spec))
        fold_count += 1

    if fold_count == 0:
        raise ValueError("Not enough labeled rows for one walk-forward fold")

    summary = {}
    for policy, folds in per_policy.items():
        summary[policy] = {
            "folds": fold_count,
            "total_trades": int(sum(item["total_trades"] for item in folds)),
            "mean_profit_factor": float(np.mean([item["profit_factor"] for item in folds])),
            "mean_expectancy": float(np.mean([item["expectancy"] for item in folds])),
            "mean_sharpe": float(np.mean([item["sharpe_ratio"] for item in folds])),
        }

    baseline = summary["side_classifier"]
    for policy in ("net_return_regression", "quantile_regression"):
        candidate = summary[policy]
        candidate["trade_change_pct"] = (
            None if baseline["total_trades"] == 0 else (candidate["total_trades"] / baseline["total_trades"] - 1.0) * 100.0
        )
        candidate["profit_factor_delta"] = candidate["mean_profit_factor"] - baseline["mean_profit_factor"]
    return {"config": config.__dict__, "policies": summary}
