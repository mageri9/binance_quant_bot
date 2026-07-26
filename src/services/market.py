import asyncio
from loguru import logger
from sqlalchemy import select

from src.config import get_settings
from src.db import AsyncSessionFactory, Kline
from src.event_bus import AsyncEventBus, CandleClosedEvent
from src.exchange.base import BaseExchange


class MarketService:
    """Служба сбора свечей с биржи и публикации CandleClosedEvent."""
    def __init__(self, bus: AsyncEventBus, exchange: BaseExchange):
        self.bus = bus
        self.exchange = exchange
        self.settings = get_settings()

    async def fetch_and_publish_klines(self, symbol: str, timeframe: str):
        try:
            df = await self.exchange.get_klines(symbol, timeframe, limit=5)
            if df.empty:
                return

            latest = df.iloc[-1]
            open_time = int(latest["open_time"])

            async with AsyncSessionFactory() as session:
                existing = await session.execute(
                    select(Kline).where(
                        Kline.symbol == symbol,
                        Kline.timeframe == timeframe,
                        Kline.open_time == open_time
                    )
                )
                if existing.scalar_one_or_none() is None:
                    kline = Kline(
                        symbol=symbol,
                        timeframe=timeframe,
                        open_time=open_time,
                        open=float(latest["open"]),
                        high=float(latest["high"]),
                        low=float(latest["low"]),
                        close=float(latest["close"]),
                        volume=float(latest["volume"])
                    )
                    session.add(kline)
                    await session.commit()

            await self.bus.publish(CandleClosedEvent(
                symbol=symbol,
                timeframe=timeframe,
                open_time=open_time,
                open=float(latest["open"]),
                high=float(latest["high"]),
                low=float(latest["low"]),
                close=float(latest["close"]),
                volume=float(latest["volume"])
            ))
        except Exception as e:
            logger.error(f"[MarketService] Ошибка скачивания свечей {symbol}: {e}")

    async def start_polling(self, interval_seconds: int = 60):
        logger.info("[MarketService] Цикл опроса свечей запущен...")
        while True:
            for symbol, timeframe in self.settings.ACTIVE_CONFIGS:
                await self.fetch_and_publish_klines(symbol, timeframe)
            await asyncio.sleep(interval_seconds)