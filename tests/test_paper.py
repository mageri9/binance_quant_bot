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

@pytest.mark.asyncio
async def test_get_closed_trades_returns_history(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)

    trade1 = await repo.create_trade(
        symbol="BTC/USDT", entry_price=100.0, amount=1.0,
        sl_price=98.0, tp_price=104.0, entry_candle_time=1000,
    )
    await repo.close_trade(trade1, exit_price=104.0, pnl=4.0)

    trade2 = await repo.create_trade(
        symbol="BTC/USDT", entry_price=110.0, amount=1.0,
        sl_price=108.0, tp_price=114.0, entry_candle_time=2000,
    )
    await repo.close_trade(trade2, exit_price=108.0, pnl=-2.0)

    closed = await repo.get_closed_trades("BTC/USDT")

    assert len(closed) == 2
    assert closed[0].entry_candle_time == 1000
    assert closed[1].entry_candle_time == 2000
    assert closed[0].pnl == 4.0
    assert closed[1].pnl == -2.0

@pytest.mark.asyncio
async def test_paper_trading_default_sl_tp(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)
    engine = PaperTradingEngine(temp_db_session)

    # Явно переопределим дефолтные настройки для проверки в рамках теста
    engine.settings.PAPER_SL_PCT = 0.015
    engine.settings.PAPER_TP_PCT = 0.035

    dummy_candles = pd.DataFrame(
        {
            "open_time": [1672531200000 + i * 3600 * 1000 for i in range(35)],
            "open": [100.0] * 35,
            "high": [100.5] * 35,
            "low": [99.5] * 35,
            "close": [100.0] * 35,
            "volume": [1000.0] * 35,
        }
    )

    predictor = MockPredictor(signal=1)

    # Вызываем БЕЗ передачи sl_pct и tp_pct
    await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=dummy_candles,
        predictor=predictor,
        horizon=5,
    )

    active_trade = await repo.get_active_trade("BTC/USDT")
    assert active_trade is not None
    # Используем pytest.approx для безопасного сравнения дробных чисел
    assert active_trade.sl_price == pytest.approx(98.5)
    assert active_trade.tp_price == pytest.approx(103.5)


@pytest.mark.asyncio
async def test_paper_trading_dynamic_position_sizing(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)
    engine = PaperTradingEngine(temp_db_session)

    # Переопределяем настройки для теста (риск 15% от баланса)
    engine.settings.PAPER_RISK_PCT = 0.15
    engine.settings.PAPER_MIN_ALLOCATION = 50.0

    # Начальный баланс портфеля = 10000.0, свободный кэш = 10000.0
    portfolio = await repo.get_portfolio()
    assert portfolio.balance == 10000.0

    dummy_candles = pd.DataFrame(
        {
            "open_time": [1672531200000 + i * 3600 * 1000 for i in range(35)],
            "open": [100.0] * 35,
            "high": [100.5] * 35,
            "low": [99.5] * 35,
            "close": [100.0] * 35,
            "volume": [1000.0] * 35,
        }
    )

    predictor = MockPredictor(signal=1)

    # Запускаем без принудительного указания объема
    msg = await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=dummy_candles,
        predictor=predictor,
        horizon=5,
    )

    assert msg is not None
    assert "Открыта виртуальная Long-позиция" in msg

    # Ожидаемый объем: 10000 * 15% = 1500.0$
    # Количество монет: 1500.0 / 100.0 = 15.0
    active_trade = await repo.get_active_trade("BTC/USDT")
    assert active_trade is not None
    assert active_trade.amount == pytest.approx(15.0)

    # Проверяем списание кэша: 10000 - 1500 = 8500
    portfolio = await repo.get_portfolio()
    assert portfolio.cash == pytest.approx(8500.0)


@pytest.mark.asyncio
async def test_paper_trading_min_allocation_warning(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)
    engine = PaperTradingEngine(temp_db_session)

    # Задаем условия: риск 1% от баланса (= 100$), но минимальный лимит сделки 200$
    engine.settings.PAPER_RISK_PCT = 0.01
    engine.settings.PAPER_MIN_ALLOCATION = 200.0

    dummy_candles = pd.DataFrame(
        {
            "open_time": [1672531200000 + i * 3600 * 1000 for i in range(35)],
            "open": [100.0] * 35,
            "high": [100.5] * 35,
            "low": [99.5] * 35,
            "close": [100.0] * 35,
            "volume": [1000.0] * 35,
        }
    )

    predictor = MockPredictor(signal=1)

    msg = await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=dummy_candles,
        predictor=predictor,
        horizon=5,
    )

    # Сделка должна отмениться, так как 100$ < 200$
    assert msg is not None
    assert "меньше минимально допустимого" in msg

    # Сделка не должна быть создана
    active_trade = await repo.get_active_trade("BTC/USDT")
    assert active_trade is None


@pytest.mark.asyncio
async def test_paper_repository_insufficient_cash_exception(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)

    # Имитируем покупку на сумму 15 000$, когда на балансе только 10 000$
    # Репозиторий должен прервать операцию и выбросить ValueError
    with pytest.raises(ValueError) as exc_info:
        await repo.create_trade(
            symbol="BTC/USDT",
            entry_price=100.0,
            amount=150.0,  # 150 монет * 100$ = 15000$
            sl_price=None,
            tp_price=None,
            entry_candle_time=1000,
        )

    assert "Недостаточно свободного кэша" in str(exc_info.value)