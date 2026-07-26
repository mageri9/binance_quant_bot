import pandas as pd
from src.exchange.base import BaseExchange


class PaperExchange(BaseExchange):
    """Локальный симулятор биржи."""
    def __init__(self, initial_balance: float = 10000.0, commission_rate: float = 0.0004):
        self.balance_free = initial_balance
        self.balance_total = initial_balance
        self.positions: dict[str, dict] = {}
        self.commission_rate = commission_rate

    async def get_balance(self) -> dict:
        return {"free": self.balance_free, "total": self.balance_total}

    async def get_position(self, symbol: str) -> dict | None:
        return self.positions.get(symbol)

    async def create_order(
        self, symbol: str, side: str, order_type: str, amount: float, price: float | None = None, reduce_only: bool = False
    ) -> dict:
        price = price or 100.0
        cost = amount * price
        comm = cost * self.commission_rate

        if not reduce_only:
            self.balance_free -= (cost + comm)
            self.positions[symbol] = {
                "symbol": symbol,
                "side": "LONG" if side.lower() == "buy" else "SHORT",
                "amount": amount,
                "entry_price": price
            }
        else:
            self.positions.pop(symbol, None)
            self.balance_free += cost - comm

        return {
            "order_id": f"paper_{int(price * 100)}",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "FILLED"
        }

    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        return pd.DataFrame()