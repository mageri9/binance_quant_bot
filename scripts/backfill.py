import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy.dialects.sqlite import insert

from src.config import get_settings
from src.db import AsyncSessionFactory, Kline, init_db
from src.exchange.binance import BinanceExchange


async def backfill(symbol: str, timeframe: str, days: int):
    await init_db()
    settings = get_settings()
    exchange = BinanceExchange(
        api_key=settings.BINANCE_API_KEY,
        secret=settings.BINANCE_API_SECRET,
        testnet=settings.TRADING_MODE != "mainnet",
    )

    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = int(since.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    logger.info(f"Fetching {symbol} ({timeframe}) candles for the last {days} days.")

    try:
        count = 0
        async with AsyncSessionFactory() as session:
            while since_ms < now_ms:
                ohlcv = await exchange.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since_ms, limit=1000
                )
                if not ohlcv:
                    break

                # Do not advance or write from a stale exchange response.
                new_candles = [candle for candle in ohlcv if int(candle[0]) >= since_ms]
                if not new_candles:
                    break

                rows = [
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "open_time": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    }
                    for candle in new_candles
                ]
                statement = insert(Kline).values(rows)
                statement = statement.on_conflict_do_update(
                    index_elements=["symbol", "timeframe", "open_time"],
                    set_={
                        "open": statement.excluded.open,
                        "high": statement.excluded.high,
                        "low": statement.excluded.low,
                        "close": statement.excluded.close,
                        "volume": statement.excluded.volume,
                    },
                )
                await session.execute(statement)
                await session.commit()
                count += len(rows)

                since_ms = int(new_candles[-1][0]) + 1

        if count:
            logger.info(f"Upserted {count} candles for {symbol}.")
        else:
            logger.warning("No candles found.")
    finally:
        await exchange.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    asyncio.run(backfill(args.symbol, args.timeframe, args.days))
