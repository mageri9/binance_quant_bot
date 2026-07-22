"""Stable Telegram rendering for structured trading notifications."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class TradingNotification:
    kind: str
    symbol: str
    side: str
    amount: Decimal | float
    price: Decimal | float
    order_id: str
    model_id: str | None = None
    confidence: float | None = None
    sl_ok: bool | None = None
    tp_ok: bool | None = None
    realized_pnl: Decimal | float | None = None
    commissions: Decimal | float | None = None
    exit_reason: str | None = None


def format_trading_notification(event: TradingNotification) -> str:
    """Render the only Telegram-facing representation of a trade event."""
    direction = "LONG" if event.side.lower() == "buy" else "SHORT"
    amount, price = float(event.amount), float(event.price)
    if event.kind == "position_opened":
        icon = "🟢" if direction == "LONG" else "🔴"
        model = f"\nМодель: <code>{event.model_id}</code>" if event.model_id else ""
        confidence = (
            f" · confidence {event.confidence:.0%}" if event.confidence is not None else ""
        )
        protection = (
            f"SL {'✅' if event.sl_ok else '⚠️'} · TP {'✅' if event.tp_ok else '⚠️'}"
            if event.sl_ok is not None or event.tp_ok is not None else "не задана"
        )
        return (
            f"{icon} <b>{event.symbol} · {direction} открыт</b>\n\n"
            f"Исполнено: <code>{amount:.6f}</code> @ <code>${price:,.2f}</code>\n"
            f"Объём: <code>${amount * price:,.2f}</code>\n"
            f"Защита: {protection}{model}{confidence}\n"
            f"Order: <code>{event.order_id}</code>"
        )
    pnl = "—" if event.realized_pnl is None else f"${float(event.realized_pnl):+,.2f}"
    fees = "—" if event.commissions is None else f"${float(event.commissions):,.2f}"
    reason = event.exit_reason or "сигнал/reconciliation"
    return (
        f"⚪️ <b>{event.symbol} · позиция закрыта</b>\n\n"
        f"Исполнено: <code>{amount:.6f}</code> @ <code>${price:,.2f}</code>\n"
        f"Фактический PnL: <code>{pnl}</code>\n"
        f"Комиссии: <code>{fees}</code>\n"
        f"Причина выхода: <code>{reason}</code>\n"
        f"Order: <code>{event.order_id}</code>"
    )
