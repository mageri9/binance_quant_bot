import ccxt.async_support as ccxt
import pandas as pd
from src.exchange.base import BaseExchange


class PaperExchange(BaseExchange):
    """Локальный симулятор биржи с получением реальных рыночных данных."""

    def __init__(
        self, initial_balance: float = 10000.0, commission_rate: float = 0.0004
    ):
        self.balance_free = initial_balance
        self.balance_total = initial_balance
        self.positions: dict[str, dict] = {}
        self.commission_rate = commission_rate
        # Публичный клиент без ключей для получения рыночной информации
        self.public_exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

    async def close(self):
        await self.public_exchange.close()

    async def get_balance(self) -> dict:
        open_positions_value = sum(
            pos["amount"] * pos["entry_price"]
            for pos in self.positions.values()
        )
        self.balance_total = self.balance_free + open_positions_value
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
            self.balance_free -= cost + comm
            self.positions[symbol] = {
                "symbol": symbol,
                "side": "LONG" if side.lower() == "buy" else "SHORT",
                "amount": amount,
                "entry_price": price,
            }
        else:
            self.positions.pop(symbol, None)
            self.balance_free += cost - comm

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