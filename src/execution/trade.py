"""Canonical trade contract shared by targets and order execution."""

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Literal

import numpy as np
import pandas as pd

from src.execution.kernel import ExecutionCosts, ExecutionKernel, SimulatedFill, costs_from_settings


Side = Literal["LONG", "SHORT"]
ExitReason = Literal["stop_loss", "take_profit", "timeout"]


@dataclass(frozen=True)
class TradePolicy:
    """All semantics that must change atomically for training and execution."""

    timeout_candles: int = 5
    sl_pct: Decimal = Decimal("0.02")
    tp_pct: Decimal = Decimal("0.04")
    costs: ExecutionCosts = ExecutionCosts()

    def __post_init__(self):
        object.__setattr__(self, "sl_pct", Decimal(str(self.sl_pct)))
        object.__setattr__(self, "tp_pct", Decimal(str(self.tp_pct)))
        if self.timeout_candles < 1:
            raise ValueError("timeout_candles must be positive")
        if self.sl_pct <= 0 or self.tp_pct <= 0:
            raise ValueError("sl_pct and tp_pct must be positive")

    def protection_prices(self, entry_reference_price: float, side: Side) -> tuple[float, float]:
        price = Decimal(str(entry_reference_price))
        if side == "LONG":
            return float(price * (1 - self.sl_pct)), float(price * (1 + self.tp_pct))
        return float(price * (1 + self.sl_pct)), float(price * (1 - self.tp_pct))

    def identity(self) -> dict:
        return {
            "entry": "next_candle_open",
            "timeout_candles": self.timeout_candles,
            "sl_pct": str(self.sl_pct),
            "tp_pct": str(self.tp_pct),
            "commission_rate": str(self.costs.commission_rate),
            "slippage_rate": str(self.costs.slippage_rate),
            "bid_ask_spread_rate": str(self.costs.bid_ask_spread_rate),
            "funding_rate_per_trade": str(self.costs.funding_rate_per_trade),
        }


@dataclass(frozen=True)
class TradeSpec:
    entry_price: float
    entry_time: int
    side: Side
    sl: float
    tp: float
    timeout: int
    commission: float
    slippage: float
    funding: float


@dataclass(frozen=True)
class TradeOutcome:
    spec: TradeSpec
    exit_price: float
    exit_reason: ExitReason
    net_return: float
    exit_time: int

    def as_record(self) -> dict:
        return {**asdict(self.spec), "exit_price": self.exit_price,
                "exit_reason": self.exit_reason, "net_return": self.net_return,
                "exit_time": self.exit_time}


def trade_policy_from_settings(settings) -> TradePolicy:
    """The only settings adapter used by both dataset and trading paths."""
    def configured(primary, legacy, default):
        value = getattr(settings, primary, None)
        if not isinstance(value, (int, float, str, Decimal)):
            value = getattr(settings, legacy, default)
        return value if isinstance(value, (int, float, str, Decimal)) else default

    return TradePolicy(
        timeout_candles=int(configured("TRADE_TIMEOUT_CANDLES", "LABEL_HORIZON", 5)),
        sl_pct=configured("TRADE_SL_PCT", "PAPER_SL_PCT", 0.02),
        tp_pct=configured("TRADE_TP_PCT", "PAPER_TP_PCT", 0.04),
        costs=costs_from_settings(settings),
    )


def evaluate_trade(df: pd.DataFrame, signal_idx: int, side: Side, policy: TradePolicy) -> TradeOutcome | None:
    """Resolve exactly the market order emitted after a closed-candle signal."""
    entry_idx = signal_idx + 1
    timeout_idx = entry_idx + policy.timeout_candles
    if entry_idx >= len(df) or timeout_idx >= len(df):
        return None

    reference = float(df.iloc[entry_idx]["open"])
    entry_time = int(df.iloc[entry_idx].get("open_time", entry_idx))
    sl, tp = policy.protection_prices(reference, side)
    kernel = ExecutionKernel(policy.costs)
    entry_fill = kernel.market_fill(
        side="buy" if side == "LONG" else "sell", reference_price=reference, amount=1
    )
    exit_reference = float(df.iloc[timeout_idx]["open"])
    exit_idx = timeout_idx
    reason: ExitReason = "timeout"

    # As in execution, a same-candle collision is resolved conservatively: SL first.
    for idx in range(entry_idx, timeout_idx + 1):
        row = df.iloc[idx]
        if (side == "LONG" and float(row["low"]) <= sl) or (
            side == "SHORT" and float(row["high"]) >= sl
        ):
            exit_reference, exit_idx, reason = sl, idx, "stop_loss"
            break
        if (side == "LONG" and float(row["high"]) >= tp) or (
            side == "SHORT" and float(row["low"]) <= tp
        ):
            exit_reference, exit_idx, reason = tp, idx, "take_profit"
            break

    exit_fill = kernel.market_fill(
        side="sell" if side == "LONG" else "buy", reference_price=exit_reference, amount=1
    )
    funding = float(entry_fill.price * policy.costs.funding_rate_per_trade)
    spec = TradeSpec(
        entry_price=float(entry_fill.price), entry_time=entry_time, side=side, sl=sl, tp=tp,
        timeout=int(df.iloc[timeout_idx].get("open_time", timeout_idx)),
        commission=float(entry_fill.commission + exit_fill.commission),
        slippage=float(policy.costs.slippage_rate), funding=funding,
    )
    return TradeOutcome(
        spec=spec, exit_price=float(exit_fill.price), exit_reason=reason,
        net_return=float(kernel.realized_return(entry=entry_fill, exit=exit_fill, is_short=side == "SHORT")),
        exit_time=int(df.iloc[exit_idx].get("open_time", exit_idx)),
    )


def build_trade_targets(df: pd.DataFrame, policy: TradePolicy) -> pd.DataFrame:
    """Build auditable examples; target is the profitable side after all costs."""
    rows = []
    for idx in range(len(df)):
        long = evaluate_trade(df, idx, "LONG", policy)
        short = evaluate_trade(df, idx, "SHORT", policy)
        if long is None or short is None:
            rows.append({"target_binary": np.nan, "target_triple": np.nan})
            continue
        chosen = long if long.net_return >= short.net_return else short
        target = 1.0 if chosen.net_return > 0 and chosen.spec.side == "LONG" else (
            -1.0 if chosen.net_return > 0 else 0.0
        )
        record = chosen.as_record()
        record.update({"target_binary": float(long.net_return > 0), "target_triple": target})
        rows.append(record)
    return pd.DataFrame(rows, index=df.index)
