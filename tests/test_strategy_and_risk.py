import pandas as pd
import numpy as np
from src.strategy.features import add_features
from src.risk.sizer import calculate_position_size, calculate_protection_prices
from src.risk.guards import RiskGuard


def test_add_features_dataframe():
    prices = np.linspace(100, 150, 50)
    df = pd.DataFrame({
        "open": prices - 0.5, "high": prices + 1.0, "low": prices - 1.0,
        "close": prices, "volume": [1000.0] * 50
    })

    df_feat = add_features(df)
    assert "rsi" in df_feat.columns
    assert "macd_pct" in df_feat.columns
    assert "atr" in df_feat.columns


def test_calculate_position_size():
    size = calculate_position_size(balance=1000.0, current_price=100.0, max_allocation_pct=0.10)
    assert size == 1.0  # 10% от $1000 = $100 -> 1 монета по $100


def test_calculate_protection_prices():
    sl, tp = calculate_protection_prices(entry_price=100.0, side="buy", sl_pct=0.02, tp_pct=0.04)
    assert sl == 98.0
    assert tp == 104.0


def test_risk_guard_circuit_breaker():
    guard = RiskGuard(consecutive_losses_limit=3)
    ok, reason = guard.validate_order("BTC/USDT", 1000.0, 0.0, consecutive_losses=3, open_positions_count=0)
    assert not ok
    assert "Circuit Breaker" in reason