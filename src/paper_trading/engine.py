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
    Выполняет сделки на основе входящих свечей и логирует их в БД.
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
        trade_allocation: float = 1000.0,
    ) -> str | None:
        """
        Основной рабочий цикл движка:
        - Проверяет открытые сделки на выход по SL / TP / Horizon.
        - При отсутствии открытой сделки опрашивает модель и открывает Long при сигнале '1'.
        Возвращает строку-лог совершенной операции.
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
            # --- Логика ведения открытой позиции ---
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

            # 3. Проверяем выход по времени (Time Horizon)
            # Берем количество свечей, прошедших с момента входа в сделку
            candles_since_entry = latest_candles[
                latest_candles["open_time"] >= active_trade.entry_candle_time
            ]
            if len(candles_since_entry) >= horizon:
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

            if signal == 1:
                portfolio = await self.repo.get_portfolio()

                # Проверяем наличие свободного кэша
                if portfolio.cash < trade_allocation:
                    msg = f"⚠️ [PAPER] Недостаточно кэша для сделки по {symbol}. Нужно: {trade_allocation}$, Свободно: {portfolio.cash:.2f}$"
                    logger.warning(msg)
                    return msg

                # Расчет объема покупки монет
                amount = trade_allocation / latest_close

                # Применяем параметры из настроек, если они не переданы явно
                effective_sl_pct = sl_pct if sl_pct is not None else self.settings.PAPER_SL_PCT
                effective_tp_pct = tp_pct if tp_pct is not None else self.settings.PAPER_TP_PCT

                sl_price = latest_close * (1.0 - effective_sl_pct) if effective_sl_pct else None
                tp_price = latest_close * (1.0 + effective_tp_pct) if effective_tp_pct else None

                await self.repo.create_trade(
                    symbol=symbol,
                    entry_price=latest_close,
                    amount=amount,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    entry_candle_time=latest_open_time,
                )

                msg = (
                    f"🚀 [PAPER] Открыта виртуальная Long-позиция по {symbol} по цене {latest_close:.2f}. "
                    f"Количество монет: {amount:.6f}. SL: {sl_price:.2f}, TP: {tp_price:.2f}"
                )
                logger.info(msg)
                return msg

            return None