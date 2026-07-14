import asyncio
from datetime import datetime, timezone
from loguru import logger
import ccxt.async_support as ccxt
from sqlalchemy.ext.asyncio import AsyncSession

from src.crud.kline import KlineRepository


class DataCollector:
    def __init__(self, session: AsyncSession):
        self.repo = KlineRepository(session)
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,
            }
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.exchange.close()

    async def fetch_and_save_klines(
        self,
        symbol: str,
        timeframe: str,
        since_datetime: datetime | None = None,
        limit: int | None = None
    ) -> int:
        """
        Скачивает свечи с Binance и сохраняет в базу данных.
        """
        if since_datetime:
            # Если часовой пояс не указан, принудительно считаем время как UTC
            if since_datetime.tzinfo is None:
                since_datetime = since_datetime.replace(tzinfo=timezone.utc)
            since_ms = int(since_datetime.timestamp() * 1000)
        else:
            since_ms = None

        logger.info(f"Запрос свечей {symbol} ({timeframe}) с {since_datetime or 'начала'}")

        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Не удалось скачать свечи с Binance: {e}")
            raise e

        if not ohlcv:
            logger.warning("Биржа вернула пустой список свечей.")
            return 0

        klines_data = []
        for candle in ohlcv:
            klines_data.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "open_time": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })

        await self.repo.save_klines(klines_data)
        logger.info(f"Успешно сохранено {len(klines_data)} свечей в базу данных.")
        return len(klines_data)