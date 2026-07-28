from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.config import Settings
from src.db import AsyncSessionFactory, Base, PredictionLog, Trade, engine
from src.event_bus import AsyncEventBus
from src.services.notifier_service import NotifierService


class DigestExchange:
    async def get_balance(self) -> dict:
        return {"total": 10042.30}

    async def get_klines(self, symbol: str, timeframe: str, limit: int = 1) -> pd.DataFrame:
        return pd.DataFrame([{"close": 61200.0}])


async def prepare_database() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)


@pytest.mark.asyncio
async def test_periodic_digest_reports_activity_and_open_position() -> None:
    await prepare_database()
    now = datetime.now(timezone.utc)
    async with AsyncSessionFactory() as session:
        session.add_all([
            Trade(
                symbol="BTC/USDT", status="CLOSED", side="LONG", entry_price=60000.0,
                exit_price=60042.3, amount=1.0, pnl=42.3, closed_at=now,
            ),
            Trade(
                symbol="BTC/USDT", status="OPEN", side="LONG", entry_price=61181.9,
                amount=1.0,
            ),
            Trade(
                symbol="ETH/USDT", status="CLOSED", side="LONG", entry_price=100.0,
                exit_price=200.0, amount=1.0, pnl=100.0, closed_at=now - timedelta(days=2),
            ),
            PredictionLog(symbol="BTC/USDT", timeframe="1h", signal=1, expected_return=0.01, price=60000.0, created_at=now),
            PredictionLog(symbol="BTC/USDT", timeframe="1h", signal=0, expected_return=0.0, price=60000.0, created_at=now),
        ])
        await session.commit()

    bot = AsyncMock()
    service = NotifierService(AsyncEventBus(), bot, DigestExchange())

    await service.send_periodic_digest()

    text = bot.send_message.await_args.kwargs["text"]
    assert "Сделки: 1 закрыто · открытых: 1" in text
    assert "PnL за период: <code>$+42.30</code>" in text
    assert "Winrate: 1/1" in text
    assert "Сигналов сгенерировано: 2 (сигналов на сделку: 1, hold: 1)" in text
    assert "BTC/USDT LONG" in text
    assert "unrealized: <code>$+18.10</code>" in text
    assert "$+100.00" not in text


@pytest.mark.asyncio
async def test_periodic_digest_is_sent_without_activity() -> None:
    await prepare_database()
    bot = AsyncMock()
    service = NotifierService(AsyncEventBus(), bot, DigestExchange())

    await service.send_periodic_digest()

    text = bot.send_message.await_args.kwargs["text"]
    assert "Сделки: 0 закрыто · открытых: 0" in text
    assert "Сигналов сгенерировано: 0 (сигналов на сделку: 0, hold: 0)" in text


def test_digest_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="DIGEST_INTERVAL_SECONDS must be positive"):
        Settings(BOT_TOKEN="token", ADMIN_IDS=[1], DIGEST_INTERVAL_SECONDS=0)
