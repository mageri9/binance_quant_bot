from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db.models import Kline


class KlineRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_klines(self, klines_data: list[dict]) -> None:
        """
        Сохраняет список свечей в базу данных.
        Если свеча с таким временем уже существует, обновляет её данные (Upsert).
        """
        if not klines_data:
            return

        for data in klines_data:
            stmt = sqlite_insert(Kline).values(
                symbol=data["symbol"],
                timeframe=data["timeframe"],
                open_time=data["open_time"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"]
            )
            # Настройка логики "если уже есть в базе — обновить цены"
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
            await self.session.execute(stmt)
        await self.session.commit()

    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> list[Kline]:
        """
        Получает последние сохраненные свечи из базы данных (от новых к старым).
        """
        stmt = (
            select(Kline)
            .where(Kline.symbol == symbol, Kline.timeframe == timeframe)
            .order_by(Kline.open_time.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())