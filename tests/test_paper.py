import asyncio

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
async def test_portfolio_balance_with_open_position(temp_db_session):
    """Проверяем, что баланс корректно отражает cash + positions_value"""
    repo = PaperTradingRepository(temp_db_session)

    # Начальный баланс
    portfolio = await repo.get_portfolio()
    assert portfolio.balance == 10000.0
    assert portfolio.cash == 10000.0
    assert portfolio.positions_value == 0.0

    # Открываем позицию на 1000$
    trade = await repo.create_trade(
        symbol="BTC/USDT",
        entry_price=100.0,
        amount=10.0,
        sl_price=90.0,
        tp_price=110.0,
        entry_candle_time=1000,
    )

    # Проверяем обновленный баланс
    portfolio = await repo.get_portfolio()
    assert portfolio.cash == 9000.0  # Уменьшился на 1000
    assert portfolio.positions_value == 1000.0  # Появилась стоимость позиции
    assert portfolio.balance == 10000.0  # Не изменился!

    # Закрываем позицию с прибылью
    await repo.close_trade(trade, exit_price=110.0, pnl=100.0)

    # Проверяем финальный баланс
    portfolio = await repo.get_portfolio()
    assert portfolio.cash == 10100.0  # 9000 + 1000 + 100
    assert portfolio.positions_value == 0.0  # Позиция закрыта
    assert portfolio.balance == 10100.0  # Обновился!


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
    # И стоимость позиции = 1500
    portfolio = await repo.get_portfolio()
    assert portfolio.cash == pytest.approx(8500.0)
    assert portfolio.positions_value == pytest.approx(1500.0)
    assert portfolio.balance == pytest.approx(10000.0)  # Не изменился!



@pytest.mark.asyncio
async def test_paper_trading_flow(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)
    engine = PaperTradingEngine(temp_db_session)

    # Явно устанавливаем риск для теста
    engine.settings.PAPER_RISK_PCT = 0.10  # 10% от баланса
    engine.settings.PAPER_MIN_ALLOCATION = 1.0

    # Изначально портфель создается со стартовыми $10000
    portfolio = await repo.get_portfolio()
    assert portfolio.balance == 10000.0
    assert portfolio.cash == 10000.0
    assert portfolio.positions_value == 0.0

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
        # Не передаем trade_allocation - используем PAPER_RISK_PCT
    )

    assert msg is not None
    assert "Открыта виртуальная Long-позиция" in msg

    # Проверяем, что $1000 списались в свободный кэш (10% от 10000)
    portfolio = await repo.get_portfolio()
    assert portfolio.cash == 9000.0  # 10000 - 1000 (10%)
    assert portfolio.positions_value == 1000.0
    assert portfolio.balance == 10000.0  # Баланс НЕ меняется до закрытия сделки

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
    assert portfolio.positions_value == 0.0
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


@pytest.mark.asyncio
async def test_paper_trading_short_tp_hit(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)
    engine = PaperTradingEngine(temp_db_session)

    # Задаем фиксированный размер сделки для простоты расчетов
    trade_allocation = 1000.0

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

    # Симулируем сигнал SHORT (-1)
    predictor = MockPredictor(signal=-1)

    # 1. Открываем SHORT
    msg = await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=dummy_candles,
        predictor=predictor,
        sl_pct=0.02,
        tp_pct=0.04,
        trade_allocation=trade_allocation,
    )

    assert msg is not None
    assert "Открыта виртуальная Short-позиция" in msg

    active_trade = await repo.get_active_trade("BTC/USDT")
    assert active_trade is not None
    # При шорте от цены входа 100.0:
    # TP должен быть ниже на 4% (96.0$)
    # SL должен быть выше на 2% (102.0$)
    assert active_trade.sl_price == pytest.approx(102.0)
    assert active_trade.tp_price == pytest.approx(96.0)

    # 2. Имитируем падение рынка (достижение TP по минимальной цене 95.0)
    drop_candles = dummy_candles.copy()
    drop_candles.loc[drop_candles.index[-1], "low"] = 95.0
    drop_candles.loc[drop_candles.index[-1], "close"] = 96.0

    close_msg = await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=drop_candles,
        predictor=predictor,
        sl_pct=0.02,
        tp_pct=0.04,
        trade_allocation=trade_allocation,
    )

    assert close_msg is not None
    assert "Сработал Take-Profit" in close_msg
    assert "SHORT" in close_msg

    # Ожидаемый PnL: (100 - 96) * 10 монет = +40.0$
    # Итоговый баланс: 10000 + 40 = 10040$
    portfolio = await repo.get_portfolio()
    assert portfolio.balance == pytest.approx(10040.0)
    assert portfolio.cash == pytest.approx(10040.0)


@pytest.mark.asyncio
async def test_paper_trading_short_sl_hit(temp_db_session):
    repo = PaperTradingRepository(temp_db_session)
    engine = PaperTradingEngine(temp_db_session)

    trade_allocation = 1000.0

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

    predictor = MockPredictor(signal=-1)

    # 1. Открываем SHORT
    await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=dummy_candles,
        predictor=predictor,
        sl_pct=0.02,
        tp_pct=0.04,
        trade_allocation=trade_allocation,
    )

    # 2. Имитируем резкий рост рынка вверх (достижение SL по максимальной цене 103.0)
    pump_candles = dummy_candles.copy()
    pump_candles.loc[pump_candles.index[-1], "high"] = 103.0
    pump_candles.loc[pump_candles.index[-1], "close"] = 102.5

    close_msg = await engine.process_market_update(
        symbol="BTC/USDT",
        timeframe="1h",
        latest_candles=pump_candles,
        predictor=predictor,
        sl_pct=0.02,
        tp_pct=0.04,
        trade_allocation=trade_allocation,
    )

    assert close_msg is not None
    assert "Сработал Stop-Loss" in close_msg

    # Ожидаемый убыток (SL на 102.0): (100 - 102) * 10 монет = -20.0$
    # Итоговый баланс: 10000 - 20 = 9980$
    portfolio = await repo.get_portfolio()
    assert portfolio.balance == pytest.approx(9980.0)
    assert portfolio.cash == pytest.approx(9980.0)


@pytest.mark.asyncio
async def test_paper_trading_timeout_exit_matches_backtest_semantics(temp_db_session):
    """
    Регрессия: движок закрывал позицию по таймауту на 1 свечу раньше,
    чем simulate_strategy (i - entry_idx >= horizon). Проверяем, что
    сделка НЕ закрывается на (horizon-1)-й свече после входа и
    закрывается ровно на horizon-й.
    """
    repo = PaperTradingRepository(temp_db_session)
    engine = PaperTradingEngine(temp_db_session)
    engine.settings.PAPER_RISK_PCT = 0.10

    base_candles = pd.DataFrame({
        "open_time": [1672531200000 + i * 3600 * 1000 for i in range(35)],
        "open": [100.0] * 35,
        "high": [100.5] * 35,
        "low": [99.5] * 35,
        "close": [100.0] * 35,
        "volume": [1000.0] * 35,
    })

    predictor = MockPredictor(signal=1)
    horizon = 3

    # Открываем позицию (SL/TP далеко, чтобы не сработали раньше времени)
    await engine.process_market_update(
        symbol="BTC/USDT", timeframe="1h",
        latest_candles=base_candles, predictor=predictor,
        sl_pct=0.5, tp_pct=0.5, horizon=horizon,
    )
    entry_trade = await repo.get_active_trade("BTC/USDT")
    entry_time = entry_trade.entry_candle_time

    # Свечи через 1 и 2 после входа (horizon-1) - сделка должна остаться открытой
    for offset in range(1, horizon):
        candles = pd.DataFrame({
            "open_time": [entry_time + i * 3600 * 1000 for i in range(-32, offset + 1)],
            "open": [100.0] * (33 + offset),
            "high": [100.5] * (33 + offset),
            "low": [99.5] * (33 + offset),
            "close": [100.0] * (33 + offset),
            "volume": [1000.0] * (33 + offset),
        })
        msg = await engine.process_market_update(
            symbol="BTC/USDT", timeframe="1h",
            latest_candles=candles, predictor=predictor,
            sl_pct=0.5, tp_pct=0.5, horizon=horizon,
        )
        assert msg is None, f"Сделка закрылась раньше времени на offset={offset}"
        assert await repo.get_active_trade("BTC/USDT") is not None

    # Свеча через horizon после входа - сделка должна закрыться
    candles = pd.DataFrame({
        "open_time": [entry_time + i * 3600 * 1000 for i in range(-32, horizon + 1)],
        "open": [100.0] * (33 + horizon),
        "high": [100.5] * (33 + horizon),
        "low": [99.5] * (33 + horizon),
        "close": [100.0] * (33 + horizon),
        "volume": [1000.0] * (33 + horizon),
    })
    msg = await engine.process_market_update(
        symbol="BTC/USDT", timeframe="1h",
        latest_candles=candles, predictor=predictor,
        sl_pct=0.5, tp_pct=0.5, horizon=horizon,
    )
    assert msg is not None
    assert "тайм-ауту" in msg
    assert await repo.get_active_trade("BTC/USDT") is None


@pytest.mark.asyncio
async def test_paper_trading_concurrent_open_no_overspend(temp_db_session):
    """
    Регрессия: без лока на портфель две параллельные задачи (символы)
    могут обе прочитать одинаковый portfolio.cash и обе пройти проверку
    достаточности средств, что даёт overspend. Эмулируем гонку явным
    конкурентным запуском process_market_update для двух разных символов
    с allocation, который в сумме превышает доступный cash, но по
    отдельности укладывается.
    """
    engine = PaperTradingEngine(temp_db_session)
    engine.settings.PAPER_MIN_ALLOCATION = 1.0

    dummy_candles = pd.DataFrame({
        "open_time": [1672531200000 + i * 3600 * 1000 for i in range(35)],
        "open": [100.0] * 35,
        "high": [100.5] * 35,
        "low": [99.5] * 35,
        "close": [100.0] * 35,
        "volume": [1000.0] * 35,
    })

    predictor = MockPredictor(signal=1)

    # Баланс 10000$. Каждая "сделка" запрашивает 6000$ -> обе по отдельности
    # проходят проверку cash < allocation (10000 >= 6000), но вместе это 12000$.
    results = await asyncio.gather(
        engine.process_market_update(
            symbol="BTC/USDT", timeframe="1h",
            latest_candles=dummy_candles, predictor=predictor,
            sl_pct=0.02, tp_pct=0.04, trade_allocation=6000.0,
        ),
        engine.process_market_update(
            symbol="ETH/USDT", timeframe="1h",
            latest_candles=dummy_candles, predictor=predictor,
            sl_pct=0.02, tp_pct=0.04, trade_allocation=6000.0,
        ),
    )

    repo = PaperTradingRepository(temp_db_session)
    portfolio = await repo.get_portfolio()

    # Ровно одна сделка должна открыться успешно, вторая - отказ по нехватке кэша
    opened = [r for r in results if r and "Открыта виртуальная" in r]
    rejected = [r for r in results if r and "Недостаточно кэша" in r]

    assert len(opened) == 1
    assert len(rejected) == 1
    # Баланс не должен уйти в минус
    assert portfolio.cash >= 0.0