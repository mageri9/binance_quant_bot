"""
Разовое наполнение БД историческими свечами с Binance.
Без этого автообучение будет копить данные по 5 свечей в час
и наберёт нужный объём только через дни.

Использование:
    docker compose exec bot python -m scripts.backfill --days 90
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from src.core.db import AsyncSessionFactory
from src.data.collector import DataCollector


async def backfill(symbol: str, timeframe: str, days: int, chunk_limit: int = 1000):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    total = 0

    async with AsyncSessionFactory() as session:
        collector = DataCollector(session)
        try:
            while since < datetime.now(timezone.utc):
                count = await collector.fetch_and_save_klines(
                    symbol=symbol, timeframe=timeframe,
                    since_datetime=since, limit=chunk_limit,
                )
                if count == 0:
                    break
                total += count
                since += timedelta(hours=chunk_limit)  # для timeframe="1h"
                if count < chunk_limit:
                    break
        finally:
            await collector.close()

    print(f"Backfill завершён: {total} свечей сохранено в БД.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    asyncio.run(backfill(args.symbol, args.timeframe, args.days))