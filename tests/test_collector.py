import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects import postgresql

from src.data.collector import DataCollector
from src.crud.kline import KlineRepository
from src.db.models import Kline


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
            since_datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
            limit=2,
        )

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


def test_kline_repository_dialect_selection():
    pg_session = MagicMock()
    pg_session.get_bind.return_value.dialect.name = "postgresql"
    repo_pg = KlineRepository(pg_session)
    assert repo_pg._insert() is pg_insert

    sqlite_session = MagicMock()
    sqlite_session.get_bind.return_value.dialect.name = "sqlite"
    repo_sqlite = KlineRepository(sqlite_session)
    assert repo_sqlite._insert() is sqlite_insert


def test_postgresql_upsert_compilation():
    stmt = pg_insert(Kline).values(
        symbol="BTC/USDT",
        timeframe="1h",
        open_time=1672531200000,
        open=16800.0,
        high=16900.0,
        low=16700.0,
        close=16850.0,
        volume=1000.0,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "timeframe", "open_time"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
        }
    )

    compiled = stmt.compile(dialect=postgresql.dialect())
    sql_str = str(compiled)

    # Проверяем ключевые блоки Postgres-синтаксиса (теперь в нижнем регистре)
    assert "INSERT INTO klines" in sql_str
    assert "ON CONFLICT (symbol, timeframe, open_time) DO UPDATE SET" in sql_str
    assert "open = excluded.open" in sql_str