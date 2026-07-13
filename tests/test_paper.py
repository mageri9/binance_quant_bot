import pytest
import pandas as pd
import numpy as np

from src.crud.paper import PaperTradingRepository
from src.paper_trading.engine import PaperTradingEngine


class MockPredictor:
    def __init__(self, signal: int):
        self.signal = signal

    def predict(self, df):
        return self.signal


@pytest.mark.asyncio
async def test_paper_trading_flow(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)
    engine = PaperTradingEngine(temp_db_session)

    # Изначально портфель создается со стартовыми $10000
    portfolio = await repo.get_portfolio()
    assert portfolio.balance == 10000.0
    assert portfolio.cash == 10000.0

    # Создаем 35 свечей для тестирования
    dummy_candles = pd.DataFrame({
        "open_time": [1672531200000 + i * 3600 * 1000 for i in range(35)],
        "open": [100.0] * 35,
        "high": [100.5] * 35,
        "low": [99.5] * 35,
        "close": [100.0] * 35,
        "volume": [1000.0] * 35
    })

    # Задаем сигнал модели на покупку (1)
    predictor = MockPredictor(signal=1)

    # Запускаем обработку рынка
    msg = await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=dummy_candles,
        predictor=predictor,
        sl_pct=0.02,
        tp_pct=0.04,
        horizon=5
    )

    assert msg is not None
    assert "Открыта виртуальная Long-позиция" in msg

    # Проверяем, что $1000 списались в свободный кэш
    portfolio = await repo.get_portfolio()
    assert portfolio.cash == 9000.0
    assert portfolio.balance == 10000.0  # Баланс не меняется до закрытия сделки

    # Проверяем параметры открытой сделки в БД
    active_trade = await repo.get_active_trade("BTC/USDT")
    assert active_trade is not None
    assert active_trade.entry_price == 100.0
    assert active_trade.amount == 10.0  # 1000$ / 100$ = 10 монет
    assert active_trade.sl_price == 98.0
    assert active_trade.tp_price == 104.0

    # Имитируем резкий обвал цены на последней свече ниже уровня SL (до 95.0)
    crash_candles = dummy_candles.copy()
    crash_candles.loc[crash_candles.index[-1], "low"] = 95.0
    crash_candles.loc[crash_candles.index[-1], "close"] = 96.0

    # Проверяем реакцию движка на обвал
    close_msg = await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=crash_candles,
        predictor=predictor,
        sl_pct=0.02,
        tp_pct=0.04,
        horizon=5
    )

    # Сделка должна закрыться по цене SL (98.0)
    assert close_msg is not None
    assert "Сработал Stop-Loss" in close_msg

    # Убыток: (98.0 - 100.0) * 10 монет = -20$
    # Возврат кэша с фиксацией убытка: 9000$ + 1000$ - 20$ = 9980$
    portfolio = await repo.get_portfolio()
    assert portfolio.cash == 9980.0
    assert portfolio.balance == 9980.0

    # В базе данных больше нет открытых позиций
    assert await repo.get_active_trade("BTC/USDT") is None