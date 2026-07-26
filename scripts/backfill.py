import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from loguru import logger
from sqlalchemy import select

from src.config import get_settings
from src.db import AsyncSessionFactory, Kline, init_db
from src.exchange.binance import BinanceExchange


async def backfill(symbol: str, timeframe: str, days: int):
    await init_db()
    settings = get_settings()

    exchange = BinanceExchange(
        api_key=settings.BINANCE_API_KEY,
        secret=settings.BINANCE_API_SECRET,
        testnet=settings.TRADING_MODE != "mainnet"
    )

    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = int(since.timestamp() * 1000)

    logger.info(f"Загрузка свечей {symbol} ({timeframe}) за последние {days} дней...")

    try:
        ohlcv = await exchange.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not ohlcv:
            logger.warning("Свечи не найдены.")
            return

        async with AsyncSessionFactory() as session:
            count = 0
            for candle in ohlcv:
                open_time = int(candle[0])
                existing = await session.execute(
                    select(Kline).where(
                        Kline.symbol == symbol, Kline.timeframe == timeframe, Kline.open_time == open_time
                    )
                )
                if existing.scalar_one_or_none() is None:
                    kline = Kline(
                        symbol=symbol,
                        timeframe=timeframe,
                        open_time=open_time,
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5])
                    )
                    session.add(kline)
                    count += 1
            await session.commit()
            logger.info(f"Сохранено {count} новых свечей {symbol} в БД.")
    finally:
        await exchange.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    asyncio.run(backfill(args.symbol, args.timeframe, args.days))