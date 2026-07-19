"""
Разовая очистка дублирующихся OPEN-сделок по symbol.
Закрывает ВСЕ открытые сделки по указанному symbol по последней цене из БД klines,
корректно пересчитывая cash/positions_value через close_trade().

Использование:
    docker compose exec bot python -m scripts.close_phantom_trades --symbol BTC/USDT --timeframe 1h
"""
import argparse
import asyncio
from sqlalchemy import select

from src.core.db import AsyncSessionFactory
from src.db.models import Trade
from src.crud.paper import TradeRepository
from src.crud.kline import KlineRepository


async def close_all_open(symbol: str, timeframe: str):
    async with AsyncSessionFactory() as session:
        stmt = select(Trade).where(
            Trade.symbol == symbol, Trade.status == "OPEN"
        )
        res = await session.execute(stmt)
        open_trades = list(res.scalars().all())

        if not open_trades:
            print(f"Нет открытых сделок по {symbol}.")
            return

        kline_repo = KlineRepository(session)
        klines = await kline_repo.get_klines(symbol, timeframe, limit=1)
        if not klines:
            print("Нет свечей для расчета текущей цены, прерываю.")
            return
        current_price = klines[0].close

        repo = TradeRepository(session)
        for trade in open_trades:
            if trade.is_short:
                pnl = (trade.entry_price - current_price) * trade.amount
            else:
                pnl = (current_price - trade.entry_price) * trade.amount
            print(
                f"Закрываю id={trade.id} "
                f"{'SHORT' if trade.is_short else 'LONG'} entry={trade.entry_price} "
                f"amount={trade.amount} по цене {current_price} PnL={pnl:.2f}"
            )
            await repo.close_trade(trade, current_price, pnl)

        portfolio = await repo.get_portfolio()
        print(f"Готово. cash={portfolio.cash:.2f} balance={portfolio.balance:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    asyncio.run(close_all_open(args.symbol, args.timeframe))