from datetime import datetime, timedelta, timezone
from html import escape

from aiogram import Bot
from loguru import logger
from sqlalchemy import func, select

from src.config import get_settings
from src.db import AsyncSessionFactory, PredictionLog, Trade
from src.event_bus import AsyncEventBus, ErrorEvent, OrderExecutedEvent, TradeClosedEvent
from src.exchange.base import BaseExchange


class NotifierService:
    """Слушает события -> Отправляет сообщения в Telegram и репортит ошибки в Nexus SDK."""
    def __init__(self, bus: AsyncEventBus, bot: Bot, exchange: BaseExchange, nexus_sdk=None):
        self.bus = bus
        self.bot = bot
        self.exchange = exchange
        self.nexus_sdk = nexus_sdk
        self.settings = get_settings()

        self.bus.subscribe(OrderExecutedEvent, self.on_order_executed)
        self.bus.subscribe(TradeClosedEvent, self.on_trade_closed)
        self.bus.subscribe(ErrorEvent, self.on_error)

    async def _send_admin_alert(self, text: str):
        for admin_id in self.settings.ADMIN_IDS:
            try:
                await self.bot.send_message(chat_id=admin_id, text=text, parse_mode="HTML")
            except Exception as e:
                logger.error(f"[NotifierService] Ошибка отправки в Telegram: {e}")

    async def send_periodic_digest(self) -> None:
        interval_seconds = self.settings.DIGEST_INTERVAL_SECONDS
        period_label = (
            f"{interval_seconds // 3600}ч"
            if interval_seconds % 3600 == 0
            else f"{interval_seconds // 60}мин"
        )
        period_start = datetime.now(timezone.utc) - timedelta(
            seconds=interval_seconds
        )
        async with AsyncSessionFactory() as session:
            closed_trades = list((await session.execute(
                select(Trade).where(
                    Trade.status == "CLOSED",
                    Trade.closed_at >= period_start,
                )
            )).scalars())
            open_trades = list((await session.execute(
                select(Trade).where(Trade.status == "OPEN")
            )).scalars())
            signals_count, trade_signals_count = (await session.execute(
                select(
                    func.count(PredictionLog.id),
                    func.count(PredictionLog.id).filter(PredictionLog.signal != 0),
                ).where(PredictionLog.created_at >= period_start)
            )).one()

        period_pnl = sum(trade.pnl or 0.0 for trade in closed_trades)
        wins = sum(1 for trade in closed_trades if (trade.pnl or 0.0) > 0)
        try:
            balance = await self.exchange.get_balance()
            current_balance = float(balance["total"])
            balance_text = f"${current_balance:,.2f}"
        except Exception as exc:
            logger.error(f"[NotifierService] Ошибка получения баланса для дайджеста: {exc}")
            balance_text = "недоступен"

        lines = [
            f"📬 <b>Дайджест MarketMind · последние {period_label}</b>",
            "",
            f"Режим: <code>{self.settings.TRADING_MODE.upper()}</code>",
            f"Баланс: <code>{balance_text}</code>",
            "",
            f"Сделки: {len(closed_trades)} закрыто · открытых: {len(open_trades)}",
            f"PnL за период: <code>${period_pnl:+,.2f}</code>",
            f"Winrate: {wins}/{len(closed_trades)}" if closed_trades else "Winrate: —",
            "",
            "Сигналов сгенерировано: "
            f"{signals_count} (сигналов на сделку: {trade_signals_count}, hold: {signals_count - trade_signals_count})",
        ]
        if open_trades:
            lines.extend(["", "<b>Открытые позиции:</b>"])
            for trade in open_trades:
                unrealized_pnl = await self._get_unrealized_pnl(trade)
                unrealized_text = (
                    f"${unrealized_pnl:+,.2f}" if unrealized_pnl is not None else "недоступен"
                )
                lines.append(
                    f"• {escape(trade.symbol)} {escape(trade.side)} @ "
                    f"<code>${trade.entry_price:,.2f}</code>, unrealized: <code>{unrealized_text}</code>"
                )

        await self._send_admin_alert("\n".join(lines))

    async def _get_unrealized_pnl(self, trade: Trade) -> float | None:
        try:
            klines = await self.exchange.get_klines(trade.symbol, "1m", limit=1)
            if klines.empty:
                return None
            current_price = float(klines.iloc[-1]["close"])
            price_change = current_price - trade.entry_price
            return price_change * trade.amount if trade.side == "LONG" else -price_change * trade.amount
        except Exception as exc:
            logger.error(f"[NotifierService] Ошибка расчета unrealized PnL для {trade.symbol}: {exc}")
            return None

    async def on_order_executed(self, event: OrderExecutedEvent):
        direction = "LONG 🟢" if event.side.lower() == "buy" else "SHORT 🔴"
        text = (
            f"🚀 <b>{event.symbol} · {direction} Открыт</b>\n\n"
            f"Объем: <code>{event.amount}</code> @ <code>${event.price:.2f}</code>\n"
            f"Stop-Loss: <code>${event.sl_price:.2f}</code>\n"
            f"Take-Profit: <code>${event.tp_price:.2f}</code>\n"
            f"Режим: <code>{event.mode.upper()}</code>"
        )
        await self._send_admin_alert(text)

    async def on_trade_closed(self, event: TradeClosedEvent):
        pnl_icon = "💰" if event.pnl >= 0 else "🔻"
        text = (
            f"⚪️ <b>{event.symbol} · Позиция Закрыта</b>\n\n"
            f"Сторона: <code>{event.side}</code>\n"
            f"Вход: <code>${event.entry_price:.2f}</code> → Выход: <code>${event.exit_price:.2f}</code>\n"
            f"{pnl_icon} PnL: <code>${event.pnl:+.2f}</code>\n"
            f"Причина: {event.reason}"
        )
        await self._send_admin_alert(text)

    async def on_error(self, event: ErrorEvent):
        text = f"🚨 <b>Критическая ошибка [{event.source}]</b>\n\n<code>{event.exception}</code>"
        await self._send_admin_alert(text)

        # Репортим ошибку в ваш Nexus SDK
        if self.nexus_sdk and hasattr(self.nexus_sdk, "report_error"):
            try:
                await self.nexus_sdk.report_error(event.exception, context=event.context)
            except Exception as exc:
                logger.error(f"[NotifierService] Nexus SDK report error failed: {exc}")
