import pytest
from src.exchange.paper import PaperExchange


@pytest.mark.asyncio
async def test_paper_exchange_execution_math(temp_db_session):
    # Настраиваем симулятор: 0.1% комиссия, 0.05% проскальзывание
    exchange = PaperExchange(
        session=temp_db_session,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )

    # Изначальный баланс = $10 000
    bal_start = await exchange.get_balance()
    assert bal_start["free"] == 10000.0

    # 1. Открываем LONG на 10 монет по базовой цене 100.0$
    # Ожидаемая цена входа с проскальзыванием: 100.0 * 1.0005 = 100.05$
    # Объем сделки: 100.05 * 10 = 1000.5$
    # Комиссия за вход: 1000.5 * 0.001 = 1.0005$
    # Ожидаемый остаток кэша: 10000 - 1000.5 (стоимость) - 1.0005 (комиссия) = 8998.4995$
    order_open = await exchange.create_order(
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        amount=10.0,
        price=100.0,
    )

    assert order_open["price"] == pytest.approx(100.05)
    assert order_open["commission"] == pytest.approx(1.0005)
    assert order_open["status"] == "open"

    bal_mid = await exchange.get_balance()
    assert bal_mid["free"] == pytest.approx(8998.4995)

    # Проверяем активную позицию в симуляторе
    pos = await exchange.get_position("BTC/USDT")
    assert pos is not None
    assert pos["side"] == "LONG"
    assert pos["entry_price"] == pytest.approx(100.05)

    # 2. Закрываем LONG встречным ордером SELL по базовой цене 110.0$
    # Ожидаемая цена выхода с проскальзыванием: 110.0 * 0.9995 = 109.945$
    # Объем выхода: 109.945 * 10 = 1099.45$
    # Комиссия за выход: 1099.45 * 0.001 = 1.09945$
    # Сырой PnL: (109.945 - 100.05) * 10 = 98.95$
    # Чистый PnL с учетом двух комиссий: 98.95 - 1.0005 (вход) - 1.09945 (выход) = 96.84995$
    # Итоговый кэш на балансе: 10000 + 96.84995 = 10096.84995$
    order_close = await exchange.create_order(
        symbol="BTC/USDT",
        side="sell",
        order_type="market",
        amount=10.0,
        price=110.0,
    )

    assert order_close["price"] == pytest.approx(109.945)
    assert order_close["commission"] == pytest.approx(1.09945)
    assert order_close["status"] == "closed"
    assert order_close["pnl"] == pytest.approx(96.84995)

    bal_end = await exchange.get_balance()
    assert bal_end["free"] == pytest.approx(10096.84995)
    assert bal_end["total"] == pytest.approx(10096.84995)

    # Позиция должна быть полностью закрыта
    assert await exchange.get_position("BTC/USDT") is None