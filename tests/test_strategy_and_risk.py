import os
import pandas as pd
import numpy as np
import pickle

from src.strategy.features import add_features
from src.strategy.model import Predictor
from src.strategy.generator import SignalGenerator
from src.risk.sizer import calculate_position_size, calculate_protection_prices
from src.risk.guards import RiskGuard


def test_add_features_dataframe():
    prices = np.linspace(100, 150, 50)
    df = pd.DataFrame({
        "open_time": np.arange(50, dtype=np.int64) * 3_600_000,
        "open": prices - 0.5, "high": prices + 1.0, "low": prices - 1.0,
        "close": prices, "volume": [1000.0] * 50
    })

    df_feat = add_features(df)
    assert "rsi" in df_feat.columns
    assert "macd_pct" in df_feat.columns
    assert "atr" in df_feat.columns
    assert all(dtype == np.dtype(np.float32) for dtype in df_feat.select_dtypes(include=[np.floating]).dtypes)
    assert df_feat["open_time"].dtype == np.dtype(np.int64)


class StubModel:
    def __init__(self, long_return: float, short_return: float) -> None:
        self.long_return = long_return
        self.short_return = short_return

    def predict_returns(self, _features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.long_return]), np.array([self.short_return])


def test_predictor_reloads_changed_model(tmp_path):
    model_path = tmp_path / "active.pkl"

    def write_model(long_return: float) -> None:
        with model_path.open("wb") as file:
            pickle.dump({"model": StubModel(long_return, 0.0), "features": ["feature"]}, file)

    write_model(0.002)
    predictor = Predictor(str(model_path))
    assert predictor.predict(pd.DataFrame({"feature": [1.0]})) == (1, 0.002)

    write_model(-0.001)
    updated_mtime = model_path.stat().st_mtime + 1
    os.utime(model_path, (updated_mtime, updated_mtime))
    assert predictor.predict(pd.DataFrame({"feature": [1.0]})) == (0, 0.0)


def test_signal_generator_uses_fallback_without_model():
    generator = SignalGenerator("missing.pkl")
    prices = np.concatenate([np.linspace(100, 70, 26), np.linspace(70, 74, 4)])
    candles = pd.DataFrame({
        "open": prices - 0.5,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": np.full(30, 1000.0),
    })

    assert generator.generate(candles) == (1, 0.002)


def test_signal_generator_loads_model_created_after_startup(tmp_path):
    model_path = tmp_path / "active.pkl"
    generator = SignalGenerator(str(model_path))
    with model_path.open("wb") as file:
        pickle.dump({"model": StubModel(0.002, 0.0), "features": ["rsi"]}, file)

    candles = pd.DataFrame({
        "open": np.arange(1, 31, dtype=float),
        "high": np.arange(2, 32, dtype=float),
        "low": np.arange(0, 30, dtype=float),
        "close": np.arange(1, 31, dtype=float),
        "volume": np.full(30, 1000.0),
    })

    assert generator.generate(candles) == (1, 0.002)


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
