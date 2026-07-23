"""Shared execution math for simulated trading environments.

Live trading records exchange-reported fills; backtests and paper trading use
this model to produce fills with the same price and cost conventions.
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ExecutionCosts:
    commission_rate: Decimal = Decimal("0.0004")
    slippage_rate: Decimal = Decimal("0.0002")
    bid_ask_spread_rate: Decimal = Decimal("0.0002")
    funding_rate_per_trade: Decimal = Decimal("0.0001")

    def __post_init__(self):
        object.__setattr__(self, "commission_rate", _decimal(self.commission_rate))
        object.__setattr__(self, "slippage_rate", _decimal(self.slippage_rate))
        object.__setattr__(self, "bid_ask_spread_rate", _decimal(self.bid_ask_spread_rate))
        object.__setattr__(self, "funding_rate_per_trade", _decimal(self.funding_rate_per_trade))


@dataclass(frozen=True)
class SimulatedFill:
    side: str
    reference_price: Decimal
    price: Decimal
    amount: Decimal
    commission: Decimal


class ExecutionKernel:
    """Applies bid/ask, adverse market-order slippage, and per-fill commission."""

    def __init__(self, costs: ExecutionCosts | None = None):
        self.costs = costs or ExecutionCosts()

    def market_fill(
        self, *, side: str, reference_price: Decimal | float, amount: Decimal | float
    ) -> SimulatedFill:
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order side: {side}")
        reference = _decimal(reference_price)
        quantity = _decimal(amount)
        half_spread = self.costs.bid_ask_spread_rate / Decimal("2")
        adverse_move = half_spread + self.costs.slippage_rate
        multiplier = Decimal("1") + adverse_move if side == "buy" else Decimal("1") - adverse_move
        price = reference * multiplier
        commission = price * quantity * self.costs.commission_rate
        return SimulatedFill(side, reference, price, quantity, commission)

    def realized_pnl(self, *, entry: SimulatedFill, exit: SimulatedFill, is_short: bool) -> Decimal:
        gross = (
            (entry.price - exit.price) * entry.amount
            if is_short
            else (exit.price - entry.price) * entry.amount
        )
        funding = entry.price * entry.amount * self.costs.funding_rate_per_trade
        return gross - entry.commission - exit.commission - funding

    def realized_return(self, *, entry: SimulatedFill, exit: SimulatedFill, is_short: bool) -> Decimal:
        notional = entry.price * entry.amount
        return Decimal("0") if notional == 0 else self.realized_pnl(entry=entry, exit=exit, is_short=is_short) / notional


def costs_from_settings(settings) -> ExecutionCosts:
    """Build the single Futures cost schedule used by backtest and paper trading."""
    return ExecutionCosts(
        commission_rate=_configured_decimal_with_legacy(
            settings, "EXECUTION_COMMISSION", "OPTUNA_COMMISSION", "0.0004"
        ),
        slippage_rate=_configured_decimal_with_legacy(
            settings, "EXECUTION_SLIPPAGE", "OPTUNA_SLIPPAGE", "0.0002"
        ),
        bid_ask_spread_rate=_configured_decimal(settings, "EXECUTION_BID_ASK_SPREAD", "0.0002"),
        funding_rate_per_trade=_configured_decimal_with_legacy(
            settings, "EXECUTION_FUNDING_PER_TRADE", "OPTUNA_FUNDING_PER_TRADE", "0.0001"
        ),
    )


def _configured_decimal(settings, name: str, default: str) -> Decimal:
    value = getattr(settings, name, default)
    return _decimal(value) if isinstance(value, (int, float, Decimal, str)) else Decimal(default)


def _configured_decimal_with_legacy(settings, name: str, legacy_name: str, default: str) -> Decimal:
    fields_set = getattr(settings, "model_fields_set", set())
    if name not in fields_set and legacy_name in fields_set:
        return _configured_decimal(settings, legacy_name, default)
    return _configured_decimal(settings, name, default)


def _decimal(value: Decimal | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))
