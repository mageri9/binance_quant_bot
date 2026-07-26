from datetime import datetime, timezone
from loguru import logger
from sqlalchemy import select

from src.config import get_settings
from src.db import AsyncSessionFactory, Trade
from src.event_bus import AsyncEventBus, OrderApprovedEvent, OrderExecutedEvent, TradeClosedEvent
from src.exchange.base import BaseExchange


class ExecutionService:
    """Слушает OrderApprovedEvent -> Исполняет на бирже -> Публикует OrderExecutedEvent."""
    def __init__(self, bus: AsyncEventBus, exchange: BaseExchange):
        self.bus = bus
        self.exchange = exchange
        self.settings = get_settings()
        self.bus.subscribe(OrderApprovedEvent, self.on_order_approved)

    async def on_order_approved(self, event: OrderApprovedEvent):
        try:
            order = await self.exchange.create_order(
                symbol=event.symbol,
                side=event.side,
                order_type="market",
                amount=event.amount,
                price=event.price
            )

            fill_price = order.get("price", event.price)
            order_id = order.get("order_id", "simulated")

            if hasattr(self.exchange, "create_stop_orders"):
                close_side = "sell" if event.side == "buy" else "buy"
                await self.exchange.create_stop_orders(
                    symbol=event.symbol,
                    side=close_side,
                    amount=event.amount,
                    sl_price=event.sl_price,
                    tp_price=event.tp_price
                )

            async with AsyncSessionFactory() as session:
                trade = Trade(
                    symbol=event.symbol,
                    status="OPEN",
                    side="LONG" if event.side.lower() == "buy" else "SHORT",
                    entry_price=fill_price,
                    amount=event.amount,
                    sl_price=event.sl_price,
                    tp_price=event.tp_price,
                    mode=self.settings.TRADING_MODE,
                    order_id=order_id
                )
                session.add(trade)
                await session.commit()

            logger.info(f"[ExecutionService] Исполнено: {event.symbol} {event.side} {event.amount} @ ${fill_price}")
            await self.bus.publish(OrderExecutedEvent(
                symbol=event.symbol,
                side=event.side,
                amount=event.amount,
                price=fill_price,
                order_id=order_id,
                sl_price=event.sl_price,
                tp_price=event.tp_price,
                mode=self.settings.TRADING_MODE
            ))
        except Exception as exc:
            logger.exception(f"[ExecutionService] Ошибка исполнения {event.symbol}: {exc}")

    async def monitor_open_trades(self):
        """Проверяет статусы активных позиций."""
        async with AsyncSessionFactory() as session:
            trades = (await session.execute(
                select(Trade).where(Trade.status == "OPEN")
            )).scalars().all()

            for trade in trades:
                pos = await self.exchange.get_position(trade.symbol)
                if pos is None or float(pos.get("amount", 0.0)) == 0.0:
                    trade.status = "CLOSED"
                    trade.closed_at = datetime.now(timezone.utc)
                    trade.exit_price = trade.tp_price or trade.entry_price
                    pnl = (trade.exit_price - trade.entry_price) * trade.amount if trade.side == "LONG" else (trade.entry_price - trade.exit_price) * trade.amount
                    trade.pnl = pnl
                    await session.commit()

                    await self.bus.publish(TradeClosedEvent(
                        symbol=trade.symbol,
                        side=trade.side,
                        amount=trade.amount,
                        entry_price=trade.entry_price,
                        exit_price=trade.exit_price,
                        pnl=pnl,
                        reason="Сработка SL/TP на бирже"
                    ))