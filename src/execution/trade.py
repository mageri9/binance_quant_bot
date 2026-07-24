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
class TradeSpec:
    """The executable trade contract shared by labels, backtests, and live orders.

    Rules are deliberately data, rather than parameters spread across callers:
    changing a barrier, timeout, or fill assumption creates a different spec.
    """

    entry_rule: str = "next_candle_open"
    sl_rule: Decimal = Decimal("0.02")
    tp_rule: Decimal = Decimal("0.04")
    timeout: int = 5
    costs: ExecutionCosts = ExecutionCosts()

    def __post_init__(self):
        object.__setattr__(self, "sl_rule", Decimal(str(self.sl_rule)))
        object.__setattr__(self, "tp_rule", Decimal(str(self.tp_rule)))
        if self.entry_rule != "next_candle_open":
            raise ValueError("only next_candle_open entry_rule is supported")
        if self.timeout < 1:
            raise ValueError("timeout must be positive")
        if self.sl_rule <= 0 or self.tp_rule <= 0:
            raise ValueError("sl_rule and tp_rule must be positive")

    @property
    def slippage(self) -> Decimal:
        """Expose the fill assumption without duplicating it beside costs."""
        return self.costs.slippage_rate

    def protection_prices(self, entry_reference_price: float, side: Side) -> tuple[float, float]:
        price = Decimal(str(entry_reference_price))
        if side == "LONG":
            return float(price * (1 - self.sl_rule)), float(price * (1 + self.tp_rule))
        return float(price * (1 + self.sl_rule)), float(price * (1 - self.tp_rule))

    def identity(self) -> dict:
        return {
            "entry_rule": self.entry_rule,
            "timeout": self.timeout,
            "sl_rule": str(self.sl_rule),
            "tp_rule": str(self.tp_rule),
            "commission_rate": str(self.costs.commission_rate),
            "slippage_rate": str(self.costs.slippage_rate),
            "bid_ask_spread_rate": str(self.costs.bid_ask_spread_rate),
            "funding_rate_per_trade": str(self.costs.funding_rate_per_trade),
        }


@dataclass(frozen=True)
class TradeRecord:
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
    record: TradeRecord
    exit_price: float
    exit_reason: ExitReason
    net_return: float
    exit_time: int

    @property
    def spec(self) -> TradeRecord:
        """Deprecated name for the executed-trade record."""
        return self.record

    def as_record(self) -> dict:
        return {**asdict(self.record), "exit_price": self.exit_price,
                "exit_reason": self.exit_reason, "net_return": self.net_return,
                "exit_time": self.exit_time}


def trade_spec_from_settings(settings) -> TradeSpec:
    """The only settings adapter used by both dataset and trading paths."""
    def configured(primary, legacy, default):
        value = getattr(settings, primary, None)
        if not isinstance(value, (int, float, str, Decimal)):
            value = getattr(settings, legacy, default)
        return value if isinstance(value, (int, float, str, Decimal)) else default

    return TradeSpec(
        timeout=int(configured("TRADE_TIMEOUT_CANDLES", "LABEL_HORIZON", 5)),
        sl_rule=configured("TRADE_SL_PCT", "PAPER_SL_PCT", 0.02),
        tp_rule=configured("TRADE_TP_PCT", "PAPER_TP_PCT", 0.04),
        costs=costs_from_settings(settings),
    )


def evaluate_trade(df: pd.DataFrame, signal_idx: int, side: Side, spec: TradeSpec) -> TradeOutcome | None:
    """Resolve exactly the market order emitted after a closed-candle signal."""
    entry_idx = signal_idx + 1
    timeout_idx = entry_idx + spec.timeout
    if entry_idx >= len(df) or timeout_idx >= len(df):
        return None

    reference = float(df.iloc[entry_idx]["open"])
    entry_time = int(df.iloc[entry_idx].get("open_time", entry_idx))
    sl, tp = spec.protection_prices(reference, side)
    kernel = ExecutionKernel(spec.costs)
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
    funding = float(entry_fill.price * spec.costs.funding_rate_per_trade)
    record = TradeRecord(
        entry_price=float(entry_fill.price), entry_time=entry_time, side=side, sl=sl, tp=tp,
        timeout=int(df.iloc[timeout_idx].get("open_time", timeout_idx)),
        commission=float(entry_fill.commission + exit_fill.commission),
        slippage=float(spec.slippage), funding=funding,
    )
    return TradeOutcome(
        record=record, exit_price=float(exit_fill.price), exit_reason=reason,
        net_return=float(kernel.realized_return(entry=entry_fill, exit=exit_fill, is_short=side == "SHORT")),
        exit_time=int(df.iloc[exit_idx].get("open_time", exit_idx)),
    )


def build_trade_targets(df: pd.DataFrame, spec: TradeSpec) -> pd.DataFrame:
    """Build auditable, cost-inclusive economic targets for both trade sides."""
    rows = []
    for idx in range(len(df)):
        long = evaluate_trade(df, idx, "LONG", spec)
        short = evaluate_trade(df, idx, "SHORT", spec)
        if long is None or short is None:
            rows.append({
                "long_net_return": np.nan,
                "short_net_return": np.nan,
                "expected_return": np.nan,
                "target_binary": np.nan,
                "target_triple": np.nan,
            })
            continue
        chosen = long if long.net_return >= short.net_return else short
        target = 1.0 if chosen.net_return > 0 and chosen.record.side == "LONG" else (
            -1.0 if chosen.net_return > 0 else 0.0
        )
        record = chosen.as_record()
        # The regression targets are the realized PnL of the exact orders the
        # execution kernel would have submitted, including every configured cost.
        record.update({
            "long_net_return": long.net_return,
            "short_net_return": short.net_return,
            "expected_return": chosen.net_return,
            # Kept only so older artifacts can still be trained and loaded.
            "target_binary": float(long.net_return > 0),
            "target_triple": target,
        })
        rows.append(record)
    return pd.DataFrame(rows, index=df.index)


# Compatibility only for artifacts/tests from the previous contract.  New code
# must construct TradeSpec and use trade_spec_from_settings directly.
class TradePolicy(TradeSpec):
    def __init__(self, *, timeout_candles=5, sl_pct=Decimal("0.02"), tp_pct=Decimal("0.04"), costs=ExecutionCosts()):
        super().__init__(timeout=timeout_candles, sl_rule=sl_pct, tp_rule=tp_pct, costs=costs)


trade_policy_from_settings = trade_spec_from_settings
