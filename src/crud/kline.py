from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db.models import Kline


class KlineRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    def _insert(self):
        if self.session.bind.dialect.name == "postgresql":
            return pg_insert
        return sqlite_insert

    async def save_klines(self, klines_data: list[dict]) -> None:
        if not klines_data:
            return

        insert_fn = self._insert()
        for data in klines_data:
            stmt = insert_fn(Kline).values(
                symbol=data["symbol"],
                timeframe=data["timeframe"],
                open_time=data["open_time"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"]
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
            await self.session.execute(stmt)
        await self.session.commit()

    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> list[Kline]:
        stmt = (
            select(Kline)
            .where(Kline.symbol == symbol, Kline.timeframe == timeframe)
            .order_by(Kline.open_time.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())