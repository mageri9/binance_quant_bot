import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from src.data.collector import DataCollector
from src.crud.kline import KlineRepository


@pytest.mark.asyncio
async def test_fetch_and_save_klines(temp_db_session):
    mock_ohlcv = [
        [1672531200000, 16800.0, 16900.0, 16700.0, 16850.0, 1000.0],
        [1672534800000, 16850.0, 17000.0, 16800.0, 16950.0, 1200.0],
    ]

    with patch("src.data.collector.ccxt.binance") as mock_binance_class:
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_ohlcv
        mock_binance_class.return_value = mock_exchange

        collector = DataCollector(temp_db_session)

        count = await collector.fetch_and_save_klines(
            symbol="BTC/USDT",
            timeframe="1h",
            # Явно указываем международный часовой пояс UTC
            since_datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
            limit=2,
        )

        # Проверяем вызов (теперь миллисекунды совпадут на любом компьютере)
        mock_exchange.fetch_ohlcv.assert_called_once_with(
            symbol="BTC/USDT", timeframe="1h", since=1672531200000, limit=2
        )
        assert count == 2

        repo = KlineRepository(temp_db_session)
        saved_klines = await repo.get_klines("BTC/USDT", "1h")
        assert len(saved_klines) == 2

        assert saved_klines[0].open_time == 1672534800000
        assert saved_klines[0].close == 16950.0
        assert saved_klines[1].open_time == 1672531200000
        assert saved_klines[1].close == 16850.0

        await collector.close()
        mock_exchange.close.assert_called_once()


@pytest.mark.asyncio
async def test_kline_upsert(temp_db_session):
    repo = KlineRepository(temp_db_session)

    initial_data = [
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "open_time": 1672531200000,
            "open": 16800.0,
            "high": 16900.0,
            "low": 16700.0,
            "close": 16850.0,
            "volume": 1000.0,
        }
    ]

    await repo.save_klines(initial_data)

    updated_data = [
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "open_time": 1672531200000,
            "open": 16800.0,
            "high": 16900.0,
            "low": 16700.0,
            "close": 16999.0,
            "volume": 1500.0,
        }
    ]

    await repo.save_klines(updated_data)

    saved_klines = await repo.get_klines("BTC/USDT", "1h")
    assert len(saved_klines) == 1
    assert saved_klines[0].close == 16999.0
    assert saved_klines[0].volume == 1500.0