import pandas as pd
from loguru import logger
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.crud.paper import PaperTradingRepository, _get_portfolio_lock


def get_timeframe_ms(timeframe: str) -> int:
    """Конвертирует строку таймфрейма в миллисекунды."""
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value * 60 * 1000
    elif unit == "h":
        return value * 60 * 60 * 1000
    elif unit == "d":
        return value * 24 * 60 * 60 * 1000
    return 3600 * 1000  # Дефолт 1 час


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
        horizon: int | None = None,
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

        # Извлекаем калиброванные параметры из артефакта модели
        model_calibration = getattr(predictor, "calibration", {})
        cal_sl = model_calibration.get("sl_pct", self.settings.PAPER_SL_PCT)
        cal_tp = model_calibration.get("tp_pct", self.settings.PAPER_TP_PCT)
        cal_hz = model_calibration.get("horizon", self.settings.LABEL_HORIZON)

        effective_horizon = horizon if horizon is not None else cal_hz

        active_trade = await self.repo.get_active_trade(symbol)

        if active_trade:
            async with _get_portfolio_lock():
                is_short = active_trade.is_short

                # Проверка таймаута сделки с SRE-защитой (резервный фолбэк для старых сделок)
                is_timeout = False
                if active_trade.timeout_candle_time is not None:
                    if latest_open_time >= active_trade.timeout_candle_time:
                        is_timeout = True
                else:
                    candles_after_entry = latest_candles[
                        latest_candles["open_time"] > active_trade.entry_candle_time
                    ]
                    if len(candles_after_entry) >= effective_horizon:
                        is_timeout = True

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

                    if active_trade.tp_price and latest_low <= active_trade.tp_price:
                        pnl = (
                            active_trade.entry_price - active_trade.tp_price
                        ) * active_trade.amount
                        await self.repo.close_trade(
                            active_trade, active_trade.tp_price, pnl
                        )
                        msg = f"🟢 [PAPER] Сработал Take-Profit по {symbol} (SHORT). Сделка закрыта по {active_trade.tp_price:.2f}. PnL: {pnl:.2f}$"
                        logger.info(msg)
                        return msg

                    if is_timeout:
                        pnl = (
                            active_trade.entry_price - latest_close
                        ) * active_trade.amount
                        await self.repo.close_trade(active_trade, latest_close, pnl)
                        msg = f"🔵 [PAPER] Выход по тайм-ауту по {symbol} (SHORT). Сделка закрыта по {latest_close:.2f}. PnL: {pnl:.2f}$"
                        logger.info(msg)
                        return msg

                else:
                    # --- ЛОГИКА ДЛЯ LONG-ПОЗИЦИИ ---
                    if active_trade.sl_price and latest_low <= active_trade.sl_price:
                        pnl = (
                            active_trade.sl_price - active_trade.entry_price
                        ) * active_trade.amount
                        await self.repo.close_trade(
                            active_trade, active_trade.sl_price, pnl
                        )
                        msg = f"🔴 [PAPER] Сработал Stop-Loss по {symbol}. Сделка закрыта по {active_trade.sl_price:.2f}. PnL: {pnl:.2f}$"
                        logger.info(msg)
                        return msg

                    if active_trade.tp_price and latest_high >= active_trade.tp_price:
                        pnl = (
                            active_trade.tp_price - active_trade.entry_price
                        ) * active_trade.amount
                        await self.repo.close_trade(
                            active_trade, active_trade.tp_price, pnl
                        )
                        msg = f"🟢 [PAPER] Сработал Take-Profit по {symbol}. Сделка закрыта по {active_trade.tp_price:.2f}. PnL: {pnl:.2f}$"
                        logger.info(msg)
                        return msg

                    if is_timeout:
                        pnl = (
                            latest_close - active_trade.entry_price
                        ) * active_trade.amount
                        await self.repo.close_trade(active_trade, latest_close, pnl)
                        msg = f"🔵 [PAPER] Выход по тайм-ауту по {symbol}. Сделка закрыта по {latest_close:.2f}. PnL: {pnl:.2f}$"
                        logger.info(msg)
                        return msg

                return None

        else:
            # --- Логика открытия новой позиции ---
            signal = predictor.predict(latest_candles)

            if signal in [1, -1]:
                async with _get_portfolio_lock():
                    portfolio = await self.repo.get_portfolio()
                    effective_trade_allocation = trade_allocation

                    if effective_trade_allocation is None:
                        effective_trade_allocation = (
                            portfolio.balance * self.settings.PAPER_RISK_PCT
                        )

                    if effective_trade_allocation < self.settings.PAPER_MIN_ALLOCATION:
                        msg = (
                            f"⚠️ [PAPER] Расчитанный объем сделки ({effective_trade_allocation:.2f}$) "
                            f"меньше минимально допустимого ({self.settings.PAPER_MIN_ALLOCATION:.2f}$)."
                        )
                        logger.warning(msg)
                        return msg

                    if portfolio.cash < effective_trade_allocation:
                        msg = (
                            f"⚠️ [PAPER] Недостаточно кэша для сделки по {symbol}. "
                            f"Нужно: {effective_trade_allocation:.2f}$, Свободно: {portfolio.cash:.2f}$"
                        )
                        logger.warning(msg)
                        return msg

                    amount = effective_trade_allocation / latest_close

                    effective_sl_pct = sl_pct if sl_pct is not None else cal_sl
                    effective_tp_pct = tp_pct if tp_pct is not None else cal_tp

                    if signal == 1:
                        sl_price = (
                            latest_close * (1.0 - effective_sl_pct)
                            if effective_sl_pct is not None
                            else None
                        )
                        tp_price = (
                            latest_close * (1.0 + effective_tp_pct)
                            if effective_tp_pct is not None
                            else None
                        )
                        pos_type = "Long"
                        is_short = False
                    else:
                        sl_price = (
                            latest_close * (1.0 + effective_sl_pct)
                            if effective_sl_pct is not None
                            else None
                        )
                        tp_price = (
                            latest_close * (1.0 - effective_tp_pct)
                            if effective_tp_pct is not None
                            else None
                        )
                        pos_type = "Short"
                        is_short = True

                    # Рассчитываем время таймаута сделки
                    timeframe_ms = get_timeframe_ms(timeframe)
                    timeout_candle_time = latest_open_time + (effective_horizon * timeframe_ms)

                    await self.repo.create_trade(
                        symbol=symbol,
                        entry_price=latest_close,
                        amount=amount,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        entry_candle_time=latest_open_time,
                        is_short=is_short,
                        timeout_candle_time=timeout_candle_time,
                    )

                sl_str = f"{sl_price:.2f}" if sl_price is not None else "-"
                tp_str = f"{tp_price:.2f}" if tp_price is not None else "-"
                msg = (
                    f"🚀 [PAPER] Открыта виртуальная {pos_type}-позиция по {symbol} по цене {latest_close:.2f}. "
                    f"Размер сделки: {effective_trade_allocation:.2f}$. "
                    f"Количество монет: {amount:.6f}. SL: {sl_str}, TP: {tp_str}"
                )
                logger.info(msg)
                return msg

            return None