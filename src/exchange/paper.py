import pandas as pd
from datetime import datetime, timezone

from src.exchange.base import BaseExchange
from src.crud.paper import PaperTradingRepository, _get_portfolio_lock
from src.crud.kline import KlineRepository


class PaperExchange(BaseExchange):
    """
    Реалистичный симулятор биржи.
    Учитывает комиссии и проскальзывание (slippage).
    """

    def __init__(
        self,
        session,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self.session = session
        self.repo = PaperTradingRepository(session)
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    async def get_balance(self) -> dict:
        portfolio = await self.repo.get_portfolio()
        return {
            "free": portfolio.cash,
            "total": portfolio.balance,
        }

    async def get_position(self, symbol: str) -> dict | None:
        trade = await self.repo.get_active_trade(symbol)
        if not trade:
            return None
        return {
            "symbol": trade.symbol,
            "side": "SHORT" if trade.is_short else "LONG",
            "entry_price": trade.entry_price,
            "amount": trade.amount,
        }

    async def get_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        kline_repo = KlineRepository(self.session)
        klines = await kline_repo.get_klines(symbol, timeframe, limit=limit)
        data = [
            {
                "open_time": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
            for k in klines
        ]
        return pd.DataFrame(data).sort_values("open_time").reset_index(drop=True)

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        side = side.lower()
        if side not in ["buy", "sell"]:
            raise ValueError("Параметр side должен быть 'buy' или 'sell'.")

        async with _get_portfolio_lock():
            if price is None:
                kline_repo = KlineRepository(self.session)
                klines = await kline_repo.get_klines(symbol, timeframe="1h", limit=1)
                if not klines:
                    raise ValueError(f"Нет доступных свечей для расчета цены {symbol}.")
                base_price = klines[0].close
            else:
                base_price = price

            # Расчет проскальзывания
            if side == "buy":
                execution_price = base_price * (1.0 + self.slippage_pct)
            else:
                execution_price = base_price * (1.0 - self.slippage_pct)

            order_value = execution_price * amount
            commission = order_value * self.commission_pct

            portfolio = await self.repo.get_portfolio()
            active_trade = await self.repo.get_active_trade(symbol)

            # Закрытие позиции встречным ордером
            if active_trade:
                is_long_close = side == "sell" and not active_trade.is_short
                is_short_close = side == "buy" and active_trade.is_short

                if is_long_close or is_short_close:
                    if is_long_close:
                        pnl = (
                            execution_price - active_trade.entry_price
                        ) * active_trade.amount
                    else:
                        pnl = (
                            active_trade.entry_price - execution_price
                        ) * active_trade.amount

                    entry_value = active_trade.entry_price * active_trade.amount
                    entry_commission = entry_value * self.commission_pct
                    real_net_pnl = pnl - entry_commission - commission

                    # Фиксируем сделку в БД
                    await self.repo.close_trade(
                        active_trade, execution_price, real_net_pnl
                    )

                    # Защита от двойного списания комиссии за вход
                    portfolio.cash += entry_commission
                    portfolio.balance = portfolio.cash + portfolio.positions_value
                    await self.session.commit()


                    return {
                        "symbol": symbol,
                        "side": side,
                        "price": execution_price,
                        "amount": amount,
                        "commission": commission,
                        "status": "closed",
                        "pnl": real_net_pnl,
                    }
                # Позиция уже открыта в ТОМ ЖЕ направлении — не пирамидим, отклоняем ордер
                return {
                    "symbol": symbol,
                    "side": side,
                    "price": execution_price,
                    "amount": amount,
                    "commission": 0.0,
                    "status": "rejected",
                    "pnl": None,
                }

            # Открытие новой позиции
            is_short = side == "sell"
            total_cost = order_value + commission
            if portfolio.cash < total_cost:
                raise ValueError(
                    f"Недостаточно кэша с учетом комиссии. "
                    f"Требуется: {total_cost:.2f}$, Доступно: {portfolio.cash:.2f}$"
                )

            entry_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            await self.repo.create_trade(
                symbol=symbol,
                entry_price=execution_price,
                amount=amount,
                sl_price=None,
                tp_price=None,
                entry_candle_time=entry_time_ms,
                is_short=is_short,
            )

            # Вычитаем комиссию за вход из кэша
            portfolio.cash -= commission
            portfolio.balance = portfolio.cash + portfolio.positions_value
            await self.session.commit()

            return {
                "symbol": symbol,
                "side": side,
                "price": execution_price,
                "amount": amount,
                "commission": commission,
                "status": "open",
                "pnl": None,
            }