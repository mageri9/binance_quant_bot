"""Canonical OOS economic-backtest contract.

Training, calibration, and promotion must use this module rather than
constructing slightly different ``simulate_strategy`` calls.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from src.execution.kernel import ExecutionCosts
from src.execution.trade import TradeSpec, trade_spec_from_settings
from src.strategy.signals import simulate_strategy


OOS_SPLIT_COLUMN = "oos_split"
CALIBRATION_SPLIT = "calibration"
ECONOMIC_TEST_SPLIT = "economic_test"


def economic_backtest_contract_from_settings(settings) -> dict:
    """Persist every execution and sizing assumption used for OOS metrics."""
    return {
        "trade_spec": trade_spec_from_settings(settings).identity(),
        "position_sizing": {
            "stop_risk_pct": _optional_number(getattr(settings, "BACKTEST_STOP_RISK_PCT", None)),
            "target_volatility": _optional_number(getattr(settings, "BACKTEST_TARGET_VOLATILITY", None)),
            "max_position_pct": _positive_number(
                getattr(settings, "BACKTEST_MAX_POSITION_PCT", 1.0), default=1.0,
            ),
        },
    }


def _optional_number(value):
    return float(value) if isinstance(value, (int, float)) else None


def _positive_number(value, *, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and value > 0 else default


def _trade_spec_from_identity(identity: dict) -> TradeSpec:
    return TradeSpec(
        entry_rule=identity["entry_rule"],
        timeout=int(identity["timeout"]),
        sl_rule=Decimal(str(identity["sl_rule"])),
        tp_rule=Decimal(str(identity["tp_rule"])),
        costs=ExecutionCosts(
            commission_rate=Decimal(str(identity["commission_rate"])),
            slippage_rate=Decimal(str(identity["slippage_rate"])),
            bid_ask_spread_rate=Decimal(str(identity["bid_ask_spread_rate"])),
            funding_rate_per_trade=Decimal(str(identity["funding_rate_per_trade"])),
        ),
    )


def simulation_kwargs(contract: dict) -> dict:
    """Build the only simulator argument set used for economic evaluation."""
    sizing = contract["position_sizing"]
    return {
        "trade_spec": _trade_spec_from_identity(contract["trade_spec"]),
        "stop_risk_pct": sizing["stop_risk_pct"],
        "target_volatility": sizing["target_volatility"],
        "max_position_pct": sizing["max_position_pct"],
    }


def contract_with_risk(contract: dict, *, horizon: int, sl_pct: float | None = None, tp_pct: float | None = None) -> dict:
    """Copy a contract while varying only the risk parameters being calibrated."""
    result = {
        "trade_spec": dict(contract["trade_spec"]),
        "position_sizing": dict(contract["position_sizing"]),
    }
    result["trade_spec"]["timeout"] = int(horizon)
    if sl_pct is not None:
        result["trade_spec"]["sl_rule"] = str(sl_pct)
    if tp_pct is not None:
        result["trade_spec"]["tp_rule"] = str(tp_pct)
    return result


def simulate_economic_backtest(df: pd.DataFrame, contract: dict, **risk_overrides) -> dict:
    """Evaluate a frame with one shared cost, sizing, and fill contract."""
    kwargs = simulation_kwargs(contract)
    kwargs.update(risk_overrides)
    return simulate_strategy(df, **kwargs)


def split_oos_partitions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return calibration and untouched economic-test OOS partitions.

    No positional fallback is allowed: it silently evaluates a different sample.
    """
    if OOS_SPLIT_COLUMN not in df.columns:
        raise ValueError(f"OOS artifact must contain '{OOS_SPLIT_COLUMN}'.")
    names = set(df[OOS_SPLIT_COLUMN].dropna())
    required = {CALIBRATION_SPLIT, ECONOMIC_TEST_SPLIT}
    if not required.issubset(names):
        raise ValueError("OOS artifact must contain calibration and economic_test partitions.")
    return (
        df.loc[df[OOS_SPLIT_COLUMN] == CALIBRATION_SPLIT].reset_index(drop=True),
        df.loc[df[OOS_SPLIT_COLUMN] == ECONOMIC_TEST_SPLIT].reset_index(drop=True),
    )


def evaluate_artifact_economic_oos(artifact: dict, df_oos: pd.DataFrame) -> dict:
    """Evaluate the artifact's persisted executable calibration on economic OOS."""
    _, economic_test = split_oos_partitions(df_oos)
    contract = artifact["economic_backtest_contract"]
    calibration = artifact.get("calibration", {})
    overrides = {}
    if calibration.get("risk_mode") == "atr":
        overrides = {
            "sl_pct": None, "tp_pct": None,
            "sl_atr_mult": calibration["sl_atr_mult"],
            "tp_atr_mult": calibration["tp_atr_mult"],
        }
    return simulate_economic_backtest(economic_test, contract, **overrides)
