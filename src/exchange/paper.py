import ccxt.async_support as ccxt
import pandas as pd
from src.exchange.base import BaseExchange


class PaperExchange(BaseExchange):
    """Локальный симулятор фьючерсной биржи с поддержкой LONG и SHORT."""

    def __init__(
        self, initial_balance: float = 10000.0, commission_rate: float = 0.0004
    ):
        self.balance_free = initial_balance
        self.balance_total = initial_balance
        self.positions: dict[str, dict] = {}
        self.commission_rate = commission_rate
        self.public_exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )

    async def close(self):
        await self.public_exchange.close()

    async def get_balance(self) -> dict:
        # Для фьючерсов: Total = Free + Margin Locked (стоимость входа) + Unrealized PnL
        locked_margin = sum(
            pos["amount"] * pos["entry_price"] for pos in self.positions.values()
        )
        self.balance_total = self.balance_free + locked_margin
        return {
            "free": self.balance_free,
            "total": self.balance_total,
        }

    async def get_position(self, symbol: str) -> dict | None:
        return self.positions.get(symbol)

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> dict:
        price = price or 100.0
        cost = amount * price
        comm = cost * self.commission_rate

        if not reduce_only:
            # Открытие позиции: блокируем маржу (cost) и снимаем комиссию
            self.balance_free -= cost + comm
            self.positions[symbol] = {
                "symbol": symbol,
                "side": "LONG" if side.lower() == "buy" else "SHORT",
                "amount": amount,
                "entry_price": price,
            }
        else:
            # Закрытие позиции: возвращаем маржу + PnL - комиссия на выход
            open_pos = self.positions.pop(symbol, None)
            if open_pos:
                entry_cost = open_pos["amount"] * open_pos["entry_price"]
                if open_pos["side"] == "LONG":
                    pnl = (price - open_pos["entry_price"]) * open_pos["amount"]
                else:  # SHORT
                    pnl = (open_pos["entry_price"] - price) * open_pos["amount"]

                # Возврат маржи + чистый PnL с учетом комиссии выхода
                self.balance_free += entry_cost + pnl - comm

        await self.get_balance()
        return {
            "order_id": f"paper_{int(price * 100)}",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "FILLED",
        }

    async def get_klines(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:
        ohlcv = await self.public_exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=limit
        )
        data = [
            {
                "open_time": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in ohlcv
        ]
        return pd.DataFrame(data)