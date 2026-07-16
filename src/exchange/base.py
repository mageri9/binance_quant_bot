from abc import ABC, abstractmethod
import pandas as pd


class BaseExchange(ABC):
    """
    Интерфейс для унифицированного взаимодействия с торговыми площадками.
    Единый стандарт для симулятора (Paper) и реального API (Binance).
    """

    @abstractmethod
    async def get_balance(self) -> dict:
        """
        Пример возвращаемого словаря:
        {"free": 1000.0, "total": 10000.0}
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> dict | None:
        """
        Пример возвращаемого словаря:
        {
            "symbol": "BTC/USDT",
            "side": "LONG",  # 'LONG' или 'SHORT'
            "entry_price": 60000.0,
            "amount": 0.1,
        }
        """
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        """
        Отправляет ордер на покупку/продажу.
        :param side: 'buy' или 'sell'
        :param order_type: 'market' или 'limit'
        """
        pass

    @abstractmethod
    async def get_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Возвращает DataFrame со столбцами:
        ['open_time', 'open', 'high', 'low', 'close', 'volume']
        """
        pass