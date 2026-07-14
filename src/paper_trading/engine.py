import pandas as pd
from loguru import logger
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.crud.paper import PaperTradingRepository
from src.db.models import PaperTrade


class PaperTradingEngine:
    """
    Движок виртуальной торговли (Paper trading).
    Поддерживает как LONG, так и SHORT позиции.
    """

    def __init__(self, session: AsyncSession):
        self.repo = PaperTradingRepository(session)
        self.settings = get_settings()

    async def process_market_update(
        self,
        symbol: str,
        timeframe: str,
        latest_candles: pd.DataFrame,
        predictor,
        sl_pct: float | None = None,
        tp_pct: float | None = None,
        horizon: int = 5,
        trade_allocation: float | None = None,
    ) -> str | None:
        """
        Основной рабочий цикл движка:
        - Проверяет открытые сделки на выход по SL / TP / Horizon.
        - При отсутствии открытой сделки опрашивает модель и открывает LONG (1) или SHORT (-1).
        """
        if len(latest_candles) < 30:
            return None

        # Считываем параметры последней закрытой свечи
        latest_candle = latest_candles.iloc[-1]
        latest_close = float(latest_candle["close"])
        latest_high = float(latest_candle["high"])
        latest_low = float(latest_candle["low"])
        latest_open_time = int(latest_candle["open_time"])

        active_trade = await self.repo.get_active_trade(symbol)

        if active_trade:
            # Читаем направление напрямую из колонки в БД без хрупких эвристик!
            is_short = active_trade.is_short

            if is_short:
                # --- ЛОГИКА ДЛЯ SHORT-ПОЗИЦИИ ---
                if active_trade.sl_price and latest_high >= active_trade.sl_price:
                    pnl = (
                        active_trade.entry_price - active_trade.sl_price
                    ) * active_trade.amount
                    await self.repo.close_trade(
                        active_trade, active_trade.sl_price, pnl
                    )
                    msg = f"🔴 [PAPER] Сработал Stop-Loss по {symbol} (SHORT). Сделка закрыта по {active_trade.sl_price:.2f}. PnL: {pnl:.2f}$"
                    logger.info(msg)
                    return msg

                # 2. Проверяем Take-Profit (для шорта это падение цены вниз)
                if active_trade.tp_price and latest_low <= active_trade.tp_price:
                    pnl = (
                        active_trade.entry_price - active_trade.tp_price
                    ) * active_trade.amount
                    await self.repo.close_trade(active_trade, active_trade.tp_price, pnl)
                    msg = f"🟢 [PAPER] Сработал Take-Profit по {symbol} (SHORT). Сделка закрыта по {active_trade.tp_price:.2f}. PnL: {pnl:.2f}$"
                    logger.info(msg)
                    return msg

                # 3. Проверяем выход по времени (Time Horizon)
                candles_after_entry = latest_candles[
                    latest_candles["open_time"] > active_trade.entry_candle_time
                ]
                # Семантика согласована с simulate_strategy: закрытие по таймауту
                # происходит ровно через `horizon` свечей ПОСЛЕ свечи входа
                # (свеча входа в счёт не идёт).
                if len(candles_after_entry) >= horizon:
                    pnl = (active_trade.entry_price - latest_close) * active_trade.amount
                    await self.repo.close_trade(active_trade, latest_close, pnl)
                    msg = f"🔵 [PAPER] Выход по тайм-ауту по {symbol} (SHORT). Сделка закрыта по {latest_close:.2f}. PnL: {pnl:.2f}$"
                    logger.info(msg)
                    return msg

            else:
                # --- ЛОГИКА ДЛЯ LONG-ПОЗИЦИИ (Базовая) ---
                # 1. Проверяем Stop-Loss
                if active_trade.sl_price and latest_low <= active_trade.sl_price:
                    pnl = (
                        active_trade.sl_price - active_trade.entry_price
                    ) * active_trade.amount
                    await self.repo.close_trade(active_trade, active_trade.sl_price, pnl)
                    msg = f"🔴 [PAPER] Сработал Stop-Loss по {symbol}. Сделка закрыта по {active_trade.sl_price:.2f}. PnL: {pnl:.2f}$"
                    logger.info(msg)
                    return msg

                # 2. Проверяем Take-Profit
                if active_trade.tp_price and latest_high >= active_trade.tp_price:
                    pnl = (
                        active_trade.tp_price - active_trade.entry_price
                    ) * active_trade.amount
                    await self.repo.close_trade(active_trade, active_trade.tp_price, pnl)
                    msg = f"🟢 [PAPER] Сработал Take-Profit по {symbol}. Сделка закрыта по {active_trade.tp_price:.2f}. PnL: {pnl:.2f}$"
                    logger.info(msg)
                    return msg

                # 3. Проверяем выход по времени
                candles_after_entry = latest_candles[
                    latest_candles["open_time"] > active_trade.entry_candle_time
                ]
                # Семантика согласована с simulate_strategy: закрытие по таймауту
                # происходит ровно через `horizon` свечей ПОСЛЕ свечи входа
                # (свеча входа в счёт не идёт).
                if len(candles_after_entry) >= horizon:
                    pnl = (latest_close - active_trade.entry_price) * active_trade.amount
                    await self.repo.close_trade(active_trade, latest_close, pnl)
                    msg = f"🔵 [PAPER] Выход по тайм-ауту по {symbol}. Сделка закрыта по {latest_close:.2f}. PnL: {pnl:.2f}$"
                    logger.info(msg)
                    return msg

            return None

        else:
            # --- Логика открытия новой позиции ---
            # Запрашиваем предсказание у модели
            signal = predictor.predict(latest_candles)

            # Signal может быть: 1 (LONG) или -1 (SHORT)
            if signal in [1, -1]:
                portfolio = await self.repo.get_portfolio()

                # Рассчитываем объем сделки
                effective_trade_allocation = trade_allocation
                if effective_trade_allocation is None:
                    effective_trade_allocation = portfolio.balance * self.settings.PAPER_RISK_PCT

                # Проверка лимитов
                if effective_trade_allocation < self.settings.PAPER_MIN_ALLOCATION:
                    msg = (
                        f"⚠️ [PAPER] Расчитанный объем сделки ({effective_trade_allocation:.2f}$) "
                        f"меньше минимально допустимого ({self.settings.PAPER_MIN_ALLOCATION:.2f}$)."
                    )
                    logger.warning(msg)
                    return msg

                # Проверяем свободный кэш
                if portfolio.cash < effective_trade_allocation:
                    msg = (
                        f"⚠️ [PAPER] Недостаточно кэша для сделки по {symbol}. "
                        f"Нужно: {effective_trade_allocation:.2f}$, Свободно: {portfolio.cash:.2f}$"
                    )
                    logger.warning(msg)
                    return msg

                amount = effective_trade_allocation / latest_close

                # Получаем калиброванные параметры напрямую из артефакта модели
                model_calibration = getattr(predictor, "calibration", {})
                cal_sl = model_calibration.get("sl_pct", self.settings.PAPER_SL_PCT)
                cal_tp = model_calibration.get("tp_pct", self.settings.PAPER_TP_PCT)

                effective_sl_pct = sl_pct if sl_pct is not None else cal_sl
                effective_tp_pct = tp_pct if tp_pct is not None else cal_tp

                # Рассчитываем уровни SL/TP в зависимости от направления
                if signal == 1:
                    sl_price = (
                        latest_close * (1.0 - effective_sl_pct)
                        if effective_sl_pct
                        else None
                    )
                    tp_price = (
                        latest_close * (1.0 + effective_tp_pct)
                        if effective_tp_pct
                        else None
                    )
                    pos_type = "Long"
                    is_short = False
                else:
                    sl_price = (
                        latest_close * (1.0 + effective_sl_pct)
                        if effective_sl_pct
                        else None
                    )
                    tp_price = (
                        latest_close * (1.0 - effective_tp_pct)
                        if effective_tp_pct
                        else None
                    )
                    pos_type = "Short"
                    is_short = True

                await self.repo.create_trade(
                    symbol=symbol,
                    entry_price=latest_close,
                    amount=amount,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    entry_candle_time=latest_open_time,
                    is_short=is_short,
                )

                msg = (
                    f"🚀 [PAPER] Открыта виртуальная {pos_type}-позиция по {symbol} по цене {latest_close:.2f}. "
                    f"Размер сделки: {effective_trade_allocation:.2f}$. "
                    f"Количество монет: {amount:.6f}. SL: {sl_price:.2f}, TP: {tp_price:.2f}"
                )
                logger.info(msg)
                return msg

            return None