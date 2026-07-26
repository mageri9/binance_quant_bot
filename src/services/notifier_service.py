from aiogram import Bot
from loguru import logger

from src.config import get_settings
from src.event_bus import AsyncEventBus, ErrorEvent, OrderExecutedEvent, TradeClosedEvent


class NotifierService:
    """Слушает события -> Отправляет сообщения в Telegram и репортит ошибки в Nexus SDK."""
    def __init__(self, bus: AsyncEventBus, bot: Bot, nexus_sdk=None):
        self.bus = bus
        self.bot = bot
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