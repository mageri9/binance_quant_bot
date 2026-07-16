import pytest
from unittest.mock import AsyncMock, patch
import pandas as pd

from src.exchange.binance import BinanceExchange


@pytest.mark.asyncio
async def test_binance_exchange_get_balance_mock():
    mock_ccxt_instance = AsyncMock()
    mock_ccxt_instance.fetch_balance.return_value = {
        "USDT": {"free": 500.0, "total": 1500.0}
    }

    with patch("ccxt.async_support.binance", return_value=mock_ccxt_instance):
        exchange = BinanceExchange(api_key="key", secret="sec", testnet=True)
        balance = await exchange.get_balance()

        assert balance["free"] == 500.0
        assert balance["total"] == 1500.0
        mock_ccxt_instance.set_sandbox_mode.assert_called_once_with(True)


@pytest.mark.asyncio
async def test_binance_exchange_get_position_mock():
    mock_ccxt_instance = AsyncMock()
    mock_ccxt_instance.fetch_positions.return_value = [
        {
            "symbol": "BTC/USDT",
            "contracts": 0.5,
            "entryPrice": 60000.0,
            "side": "long",
        }
    ]

    with patch("ccxt.async_support.binance", return_value=mock_ccxt_instance):
        exchange = BinanceExchange(api_key="key", secret="sec", testnet=True)
        pos = await exchange.get_position("BTC/USDT")

        assert pos is not None
        assert pos["symbol"] == "BTC/USDT"
        assert pos["side"] == "LONG"
        assert pos["amount"] == 0.5
        assert pos["entry_price"] == 60000.0


@pytest.mark.asyncio
async def test_binance_exchange_get_klines_mock():
    mock_ccxt_instance = AsyncMock()
    mock_ccxt_instance.fetch_ohlcv.return_value = [
        [1672531200000, 16800.0, 16900.0, 16700.0, 16850.0, 1000.0]
    ]

    with patch("ccxt.async_support.binance", return_value=mock_ccxt_instance):
        exchange = BinanceExchange(api_key="key", secret="sec", testnet=True)
        df = await exchange.get_klines("BTC/USDT", "1h", 1)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["close"].iloc[0] == 16850.0