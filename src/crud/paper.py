from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from src.db.models import PaperPortfolio, PaperTrade


class PaperTradingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_portfolio(self) -> PaperPortfolio:
        """
        Загружает данные портфеля. Если портфель пуст, создаёт новый с балансом $10 000.
        """
        stmt = select(PaperPortfolio).limit(1)
        res = await self.session.execute(stmt)
        portfolio = res.scalar_one_or_none()

        if not portfolio:
            portfolio = PaperPortfolio(balance=10000.0, cash=10000.0)
            self.session.add(portfolio)
            await self.session.commit()
            await self.session.refresh(portfolio)

        return portfolio

    async def get_active_trade(self, symbol: str) -> PaperTrade | None:
        """
        Возвращает текущую открытую сделку по паре.
        """
        stmt = (
            select(PaperTrade)
            .where(PaperTrade.symbol == symbol, PaperTrade.status == "OPEN")
            .limit(1)
        )
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()

    async def create_trade(
        self,
        symbol: str,
        entry_price: float,
        amount: float,
        sl_price: float | None,
        tp_price: float | None,
        entry_candle_time: int,
    ) -> PaperTrade:
        """
        Открывает новую виртуальную сделку и списывает средства со свободного кэша.
        """
        portfolio = await self.get_portfolio()
        cost = entry_price * amount

        # Проверка на наличие достаточного количества свободного кэша
        if portfolio.cash < cost:
            raise ValueError(
                f"Недостаточно свободного кэша в портфеле. Требуется: {cost:.2f}$, Доступно: {portfolio.cash:.2f}$"
            )

        # Списываем свободный кэш портфеля
        portfolio.cash -= cost

        trade = PaperTrade(
            symbol=symbol,
            status="OPEN",
            entry_price=entry_price,
            amount=amount,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_candle_time=entry_candle_time,
        )
        self.session.add(trade)
        await self.session.commit()
        await self.session.refresh(trade)
        return trade

    async def close_trade(
        self, trade: PaperTrade, exit_price: float, pnl: float
    ) -> None:
        """
        Закрывает сделку, возвращает кэш обратно на баланс и фиксирует финансовый результат.
        """
        trade.status = "CLOSED"
        trade.exit_price = exit_price
        trade.exit_time = datetime.now(timezone.utc)
        trade.pnl = pnl

        portfolio = await self.get_portfolio()
        # Возвращаем стоимость позиции и прибавляем PnL (профит / лосс)
        portfolio.cash += (trade.entry_price * trade.amount) + pnl
        portfolio.balance += pnl

        await self.session.commit()

    async def get_closed_trades(
        self, symbol: str, limit: int = 500
    ) -> list[PaperTrade]:
        """
        Возвращает историю закрытых сделок по паре (от старых к новым).
        Используется для расчёта стратегических метрик (Sharpe, Drawdown и т.д.)
        по фактическим результатам paper trading, а не по бэктесту.
        """
        stmt = (
            select(PaperTrade)
            .where(PaperTrade.symbol == symbol, PaperTrade.status == "CLOSED")
            .order_by(PaperTrade.entry_candle_time.asc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())