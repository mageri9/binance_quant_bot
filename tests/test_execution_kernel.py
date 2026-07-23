from decimal import Decimal

import pytest
import pandas as pd

from src.execution.kernel import ExecutionCosts, ExecutionKernel
from src.strategy.signals import simulate_strategy


def test_market_fills_apply_adverse_slippage_and_commission():
    kernel = ExecutionKernel(ExecutionCosts(
        commission_rate=Decimal("0.001"), slippage_rate=Decimal("0.002"),
        funding_rate_per_trade=Decimal("0"),
    ))

    buy = kernel.market_fill(side="buy", reference_price=100, amount=2)
    sell = kernel.market_fill(side="sell", reference_price=110, amount=2)

    assert buy.price == Decimal("100.200")
    assert sell.price == Decimal("109.780")
    assert kernel.realized_pnl(entry=buy, exit=sell, is_short=False) == Decimal("18.740040")


def test_backtest_uses_execution_kernel_for_trade_return():
    kernel = ExecutionKernel(ExecutionCosts(
        commission_rate=Decimal("0.001"), slippage_rate=Decimal("0.002"),
        funding_rate_per_trade=Decimal("0.0005"),
    ))
    metrics = simulate_strategy(
        pd.DataFrame({
            "close": [100.0, 100.0, 110.0],
            "high": [100.0, 100.0, 110.0],
            "low": [100.0, 100.0, 110.0],
            "predicted_signal": [1, 0, 0],
        }),
        horizon=1, sl_pct=None, tp_pct=None, execution_kernel=kernel,
    )
    entry = kernel.market_fill(side="buy", reference_price=100, amount=1)
    exit = kernel.market_fill(side="sell", reference_price=100, amount=1)
    assert metrics["total_trades"] == 1
    assert metrics["total_return"] == pytest.approx(
        float(kernel.realized_return(entry=entry, exit=exit, is_short=False))
    )
