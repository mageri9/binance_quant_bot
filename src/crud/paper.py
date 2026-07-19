from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from weakref import WeakKeyDictionary
import asyncio

from src.db.models import Portfolio, Trade, PredictionLog


class AsyncRLock:
    """
    Реентерабельный асинхронный лок (Reentrant Lock).
    Позволяет одной и той же задаче повторно захватывать блокировку без дедлока.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner = None
        self._count = 0

    async def acquire(self):
        me = asyncio.current_task()
        if self._owner == me:
            self._count += 1
            return
        await self._lock.acquire()
        self._owner = me
        self._count = 1

    def release(self):
        me = asyncio.current_task()
        if self._owner != me:
            raise RuntimeError("Нельзя освободить блокировку, принадлежащую другой задаче")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Общий на процесс лок для синхронизации операций с балансом портфеля.
_portfolio_locks: "WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncRLock]" = WeakKeyDictionary()


def _get_portfolio_lock() -> AsyncRLock:
    """Возвращает реентерабельный лок, привязанный к текущему запущенному event loop."""
    loop = asyncio.get_running_loop()
    lock = _portfolio_locks.get(loop)
    if lock is None:
        lock = AsyncRLock()
        _portfolio_locks[loop] = lock
    return lock


class TradeRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_portfolio(self) -> Portfolio:
        """
        Загружает данные портфеля. Если портфель пуст, создаёт новый с балансом $10 000.
        Использует Double-Checked Locking с AsyncRLock для безопасной инициализации.
        """
        stmt = select(Portfolio).limit(1)
        res = await self.session.execute(stmt)
        portfolio = res.scalar_one_or_none()

        if not portfolio:
            async with _get_portfolio_lock():
                # Повторная проверка под локом
                res2 = await self.session.execute(select(Portfolio).limit(1))
                portfolio = res2.scalar_one_or_none()

                if not portfolio:
                    portfolio = Portfolio(
                        balance=10000.0, cash=10000.0, positions_value=0.0
                    )
                    self.session.add(portfolio)
                    await self.session.commit()
                    await self.session.refresh(portfolio)

        # Пересчитываем баланс на основе cash + positions_value
        portfolio.balance = portfolio.cash + portfolio.positions_value

        return portfolio

    async def get_all_open_trades(self) -> list[Trade]:
        """Возвращает все открытые сделки"""
        stmt = (
            select(Trade)
            .where(Trade.status == "OPEN")
            .order_by(Trade.entry_candle_time)
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def get_active_trade(self, symbol: str) -> Trade | None:
        """
        Возвращает текущую открытую сделку по паре.
        """
        stmt = (
            select(Trade)
            .where(Trade.symbol == symbol, Trade.status == "OPEN")
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
        is_short: bool = False,
        timeout_candle_time: int | None = None,
    ) -> Trade:
        """
        Открывает новую сделку и фиксирует изменения в локальном кэше баланса.
        Выполняется строго под реентерабельным локом портфеля.
        """
        async with _get_portfolio_lock():
            existing = await self.get_active_trade(symbol)
            if existing is not None:
                raise ValueError(
                    f"Попытка открыть вторую позицию по {symbol} при уже открытой "
                    f"сделке id={existing.id}. Операция отклонена."
                )

            portfolio = await self.get_portfolio()
            cost = entry_price * amount

            portfolio.cash -= cost
            portfolio.positions_value += cost
            portfolio.balance = portfolio.cash + portfolio.positions_value

            trade = Trade(
                symbol=symbol,
                status="OPEN",
                entry_price=entry_price,
                amount=amount,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_candle_time=entry_candle_time,
                is_short=is_short,
                timeout_candle_time=timeout_candle_time,
            )
            self.session.add(trade)
            await self.session.commit()
            await self.session.refresh(trade)
            return trade

    async def close_trade(
        self, trade: Trade, exit_price: float, pnl: float
    ) -> None:
        """
        Закрывает сделку, возвращает кэш обратно на баланс и фиксирует финансовый результат.
        Выполняется строго под реентерабельным локом портфеля с проверкой идемпотентности.
        """
        async with _get_portfolio_lock():
            # Защита от повторного закрытия (идемпотентность)
            if trade.status != "OPEN":
                from loguru import logger
                logger.warning(
                    f"[TradeRepository] Попытка повторного закрытия сделки id={trade.id}. Операция проигнорирована."
                )
                return

            trade.status = "CLOSED"
            trade.exit_price = exit_price
            trade.exit_time = datetime.now(timezone.utc)
            trade.pnl = pnl

            portfolio = await self.get_portfolio()

            # Получаем стоимость позиции
            position_value = trade.entry_price * trade.amount

            # Возвращаем стоимость позиции и прибавляем PnL
            portfolio.cash += position_value + pnl

            # Убираем стоимость позиции
            portfolio.positions_value -= position_value

            # Обновляем баланс
            portfolio.balance = portfolio.cash + portfolio.positions_value

            await self.session.commit()

    async def get_closed_trades(
        self, symbol: str, limit: int = 500
    ) -> list[Trade]:
        """
        Возвращает историю закрытых сделок по паре (от старых к новым).
        """
        stmt = (
            select(Trade)
            .where(Trade.symbol == symbol, Trade.status == "CLOSED")
            .order_by(Trade.entry_candle_time.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        return list(reversed(res.scalars().all()))

    async def log_prediction(
        self,
        symbol: str,
        model_id: str,
        price: float,
        prediction: int,
        prob_short: float,
        prob_hold: float,
        prob_long: float,
    ) -> PredictionLog:
        """
        Записывает прогноз модели и вероятности классов в базу данных для MLOps-мониторинга.
        """
        log_entry = PredictionLog(
            symbol=symbol,
            model_id=model_id,
            price=price,
            prediction=prediction,
            prob_short=prob_short,
            prob_hold=prob_hold,
            prob_long=prob_long,
        )
        self.session.add(log_entry)
        await self.session.commit()
        return log_entry