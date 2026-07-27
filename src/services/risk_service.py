import asyncio
from datetime import datetime, timezone, timedelta
from loguru import logger
from sqlalchemy import select

from src.config import get_settings
from src.db import AsyncSessionFactory, Trade
from src.event_bus import AsyncEventBus, SignalEmittedEvent, OrderApprovedEvent
from src.exchange.base import BaseExchange
from src.risk.guards import RiskGuard
from src.risk.sizer import calculate_position_size, calculate_protection_prices


class RiskService:
    """Слушает SignalEmittedEvent -> Проверяет лимиты -> Публикует OrderApprovedEvent."""
    def __init__(self, bus: AsyncEventBus, exchange: BaseExchange):
        self.bus = bus
        self.exchange = exchange
        self.settings = get_settings()
        self.guard = RiskGuard(
            max_daily_loss_pct=self.settings.RISK_MAX_DAILY_LOSS_PCT,
            consecutive_losses_limit=self.settings.RISK_CONSECUTIVE_LOSSES_LIMIT
        )
        self._pending_symbols: set[str] = set()
        self._pending_symbols_lock = asyncio.Lock()
        self.bus.subscribe(SignalEmittedEvent, self.on_signal)

    async def on_signal(self, event: SignalEmittedEvent):
        async with self._pending_symbols_lock:
            if event.symbol in self._pending_symbols:
                logger.warning(f"[RiskService] Signal {event.symbol} rejected: order is pending")
                return
            order = await self._approve_signal(event)
            if order is None:
                return
            self._pending_symbols.add(event.symbol)

        try:
            await self.bus.publish(order)
        finally:
            async with self._pending_symbols_lock:
                self._pending_symbols.discard(event.symbol)

    async def _approve_signal(self, event: SignalEmittedEvent) -> OrderApprovedEvent | None:
        async with AsyncSessionFactory() as session:
            since_24h = datetime.now(timezone.utc) - timedelta(hours=24)
            closed_res = await session.execute(
                select(Trade).where(Trade.closed_at >= since_24h, Trade.status == "CLOSED")
            )
            daily_pnl = sum(t.pnl or 0.0 for t in closed_res.scalars().all())

            all_closed = (await session.execute(
                select(Trade).where(Trade.status == "CLOSED").order_by(Trade.closed_at.desc()).limit(10)
            )).scalars().all()
            consecutive_losses = 0
            for t in all_closed:
                if t.pnl and t.pnl < 0:
                    consecutive_losses += 1
                else:
                    break

            open_positions = (await session.execute(
                select(Trade).where(Trade.status == "OPEN")
            )).scalars().all()

            closing_trade = next(
                (
                    trade for trade in open_positions
                    if trade.symbol == event.symbol and (
                        (event.signal == -1 and trade.side == "LONG")
                        or (event.signal == 1 and trade.side == "SHORT")
                    )
                ),
                None,
            )
            is_closing = closing_trade is not None

            already_open = any(t.symbol == event.symbol for t in open_positions)
            if already_open and not is_closing:
                logger.warning(f"[RiskService] Position for {event.symbol} is already open. Rejecting signal.")
                return None

            balance = await self.exchange.get_balance()

            approved, reason = self.guard.validate_order(
                symbol=event.symbol,
                balance_total=balance["total"],
                daily_pnl=daily_pnl,
                consecutive_losses=consecutive_losses,
                open_positions_count=len(open_positions),
                is_closing=is_closing
            )

            if not approved:
                logger.warning(f"[RiskService] Сигнал {event.symbol} отклонен: {reason}")
                return

            side = "buy" if event.signal == 1 else "sell"
            amount = (
                closing_trade.amount
                if closing_trade is not None
                else calculate_position_size(
                    balance=balance["free"],
                    current_price=event.close_price,
                    max_allocation_pct=self.settings.RISK_MAX_ALLOCATION_PCT,
                )
            )

            if amount <= 0:
                return

            sl_price, tp_price = calculate_protection_prices(
                entry_price=event.close_price,
                side=side,
                sl_pct=self.settings.DEFAULT_SL_PCT,
                tp_pct=self.settings.DEFAULT_TP_PCT
            )

            logger.info(f"[RiskService] Ордер ОДОБРЕН {event.symbol}: {side.upper()} {amount} @ {event.close_price}")
            return OrderApprovedEvent(
                symbol=event.symbol,
                side=side,
                amount=amount,
                price=event.close_price,
                sl_price=sl_price,
                tp_price=tp_price,
                reason=reason,
                is_closing=is_closing,
            )
