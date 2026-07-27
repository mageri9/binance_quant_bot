import os
import pandas as pd
import numpy as np
import pickle
import pytest

from src.strategy.features import add_features
from src.strategy.model import Predictor
from src.strategy.generator import SignalGenerator
from src.risk.sizer import calculate_position_size, calculate_protection_prices
from src.risk.guards import RiskGuard
from src.services.execution_service import ExecutionService
from src.event_bus import AsyncEventBus
from src.exchange.paper import PaperExchange


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
    assert "bb_middle_pct" in df_feat.columns
    assert np.isclose(
        df_feat["bb_middle_pct"].iloc[-1],
        (prices[-1] - prices[-20:].mean()) / prices[-1],
    )
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
    assert predictor.predict(pd.DataFrame({"other_feature": [1.0]})) == (0, 0.0)

    write_model(-0.001)
    updated_mtime = model_path.stat().st_mtime + 1
    os.utime(model_path, (updated_mtime, updated_mtime))
    assert predictor.predict(pd.DataFrame({"feature": [1.0]})) == (0, 0.0)


def test_signal_generator_uses_fallback_without_model(tmp_path):
    generator = SignalGenerator(str(tmp_path))
    prices = np.concatenate([np.linspace(100, 70, 26), np.linspace(70, 74, 4)])
    candles = pd.DataFrame({
        "open": prices - 0.5,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": np.full(30, 1000.0),
    })

    assert generator.generate(candles) == (1, 0.002)


def test_signal_generator_resolves_models_per_symbol(tmp_path):
    generator = SignalGenerator(str(tmp_path))
    btc_model_path = tmp_path / "lgbm_BTCUSDT_1h.pkl"
    eth_model_path = tmp_path / "lgbm_ETHUSDT_1h.pkl"
    futures_model_path = tmp_path / "lgbm_BTCUSDTUSDT_1h.pkl"
    with btc_model_path.open("wb") as file:
        pickle.dump({"model": StubModel(0.002, 0.0), "features": ["rsi"]}, file)
    with eth_model_path.open("wb") as file:
        pickle.dump({"model": StubModel(0.0, 0.003), "features": ["rsi"]}, file)
    with futures_model_path.open("wb") as file:
        pickle.dump({"model": StubModel(0.004, 0.0), "features": ["rsi"]}, file)

    candles = pd.DataFrame({
        "open": np.arange(1, 31, dtype=float),
        "high": np.arange(2, 32, dtype=float),
        "low": np.arange(0, 30, dtype=float),
        "close": np.arange(1, 31, dtype=float),
        "volume": np.full(30, 1000.0),
    })

    assert generator.generate(candles, symbol="BTC/USDT", timeframe="1h") == (1, 0.002)
    assert generator.generate(candles, symbol="ETH/USDT", timeframe="1h") == (-1, 0.003)
    assert generator.generate(candles, symbol="BTC/USDT:USDT", timeframe="1h") == (1, 0.004)
    assert set(generator.predictors) == {
        ("BTC/USDT", "1h"),
        ("ETH/USDT", "1h"),
        ("BTC/USDT:USDT", "1h"),
    }


def test_signal_generator_loads_model_created_after_startup(tmp_path):
    generator = SignalGenerator(str(tmp_path))
    prices = np.concatenate([np.linspace(100, 70, 26), np.linspace(70, 74, 4)])
    candles = pd.DataFrame({
        "open": prices - 0.5,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": np.full(30, 1000.0),
    })

    # A missing ETH model must use its own fallback even after another key is cached.
    btc_model_path = tmp_path / "lgbm_BTCUSDT_1h.pkl"
    with btc_model_path.open("wb") as file:
        pickle.dump({"model": StubModel(0.0, 0.003), "features": ["rsi"]}, file)
    assert generator.generate(candles, symbol="BTC/USDT", timeframe="1h") == (-1, 0.003)
    assert generator.generate(candles, symbol="ETH/USDT", timeframe="1h") == (1, 0.002)
    assert ("ETH/USDT", "1h") not in generator.predictors

    eth_model_path = tmp_path / "lgbm_ETHUSDT_1h.pkl"
    with eth_model_path.open("wb") as file:
        pickle.dump({"model": StubModel(0.0, 0.003), "features": ["rsi"]}, file)

    assert generator.generate(candles, symbol="ETH/USDT", timeframe="1h") == (-1, 0.003)
    assert ("ETH/USDT", "1h") in generator.predictors


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


@pytest.mark.asyncio
async def test_exit_price_uses_stop_loss_for_losing_long_trade():
    class Exchange(PaperExchange):
        async def get_klines(self, *_args, **_kwargs):
            return pd.DataFrame([{"close": 95.0}])

    service = ExecutionService(AsyncEventBus(), Exchange())
    trade = type("Trade", (), {"symbol": "BTC/USDT", "entry_price": 100.0, "side": "LONG", "sl_price": 98.0, "tp_price": 104.0})()
    assert await service._get_exit_price(trade) == 98.0
