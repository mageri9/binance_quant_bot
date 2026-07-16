import pytest
from src.exchange.base import BaseExchange


def test_base_exchange_cannot_be_instantiated():
    # Напрямую создавать абстрактный класс запрещено
    with pytest.raises(TypeError):
        BaseExchange()


def test_base_exchange_subclass_success():
    # Правильный подкласс должен компилироваться без ошибок
    class MockExchange(BaseExchange):
        async def get_balance(self) -> dict:
            return {"free": 1000.0, "total": 10000.0}

        async def get_position(self, symbol: str) -> dict | None:
            return None

        async def create_order(
            self,
            symbol: str,
            side: str,
            order_type: str,
            amount: float,
            price: float | None = None,
        ) -> dict:
            return {}

        async def get_klines(self, symbol: str, timeframe: str, limit: int) -> any:
            return None

    obj = MockExchange()
    assert isinstance(obj, BaseExchange)