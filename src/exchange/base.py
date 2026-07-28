from abc import ABC, abstractmethod
import pandas as pd


class BaseExchange(ABC):
    @abstractmethod
    async def get_balance(self) -> dict:
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> dict | None:
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        reduce_only: bool = False
    ) -> dict:
        pass

    async def create_stop_orders(
        self, symbol: str, position_side: str, amount: float, sl_price: float, tp_price: float
    ) -> dict | None:
        """Create native protective orders when the exchange supports them."""
        return None

    @abstractmethod
    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        pass
