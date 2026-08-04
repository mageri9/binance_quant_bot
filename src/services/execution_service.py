from datetime import datetime, timezone
from loguru import logger
from sqlalchemy import select

from src.config import get_settings
from src.db import AsyncSessionFactory, Trade
from src.event_bus import (
    AsyncEventBus,
    ErrorEvent,
    OrderApprovedEvent,
    OrderExecutedEvent,
    TradeClosedEvent,
)
from src.exchange.base import BaseExchange
from src.exchange.paper import PaperExchange


class ExecutionService:
    """Слушает OrderApprovedEvent -> Исполняет на бирже -> Публикует OrderExecutedEvent."""

    def __init__(self, bus: AsyncEventBus, exchange: BaseExchange):
        self.bus = bus
        self.exchange = exchange
        self.settings = get_settings()
        self.bus.subscribe(OrderApprovedEvent, self.on_order_approved)

    async def sync_paper_state(self) -> None:
        """Синхронизирует состояние PaperExchange с SQLite при старте бота."""
        if not isinstance(self.exchange, PaperExchange):
            return

        async with AsyncSessionFactory() as session:
            open_trades = list(
                (await session.execute(
                    select(Trade).where(Trade.status == "OPEN")
                )).scalars()
            )
            closed_trades = list(
                (await session.execute(
                    select(Trade).where(Trade.status == "CLOSED")
                )).scalars()
            )

        total_realized_pnl = sum(t.pnl or 0.0 for t in closed_trades)
        comm_rate = getattr(self.exchange, "commission_rate", 0.0004)
        total_commissions = sum(
            (t.amount * t.entry_price + t.amount * (t.exit_price or t.entry_price)) * comm_rate
            for t in closed_trades
        )

        balance_free = 10000.0 + total_realized_pnl - total_commissions

        positions = {}
        for t in open_trades:
            cost = t.amount * t.entry_price
            comm = cost * comm_rate
            balance_free -= cost + comm
            positions[t.symbol] = {
                "symbol": t.symbol,
                "side": t.side,
                "amount": t.amount,
                "entry_price": t.entry_price,
            }

        self.exchange.balance_free = max(0.0, balance_free)
        self.exchange.positions = positions
        await self.exchange.get_balance()
        logger.info(
            f"[ExecutionService] Paper state synced with DB. "
            f"Free: ${self.exchange.balance_free:.2f}, Total: ${self.exchange.balance_total:.2f}, Open positions: {len(positions)}"
        )

    async def on_order_approved(self, event: OrderApprovedEvent):
        try:
            order = await self.exchange.create_order(
                symbol=event.symbol,
                side=event.side,
                order_type="market",
                amount=event.amount,
                price=event.price,
                reduce_only=event.is_closing,
            )

            fill_price = order.get("price", event.price)
            order_id = order.get("order_id", "simulated")

            async with AsyncSessionFactory() as session:
                if event.is_closing:
                    open_trade = (
                        await session.execute(
                            select(Trade)
                            .where(
                                Trade.symbol == event.symbol,
                                Trade.status == "OPEN",
                                Trade.side == ("SHORT" if event.side.lower() == "buy" else "LONG"),
                            )
                            .order_by(Trade.created_at.desc())
                            .limit(1)
                        )
                    ).scalar_one_or_none()
                    if open_trade is None:
                        raise RuntimeError(f"No open trade to close for {event.symbol}")
                    open_trade.status = "CLOSED"
                    open_trade.closed_at = datetime.now(timezone.utc)
                    open_trade.exit_price = fill_price
                    open_trade.pnl = (
                        (fill_price - open_trade.entry_price) * open_trade.amount
                        if open_trade.side == "LONG"
                        else (open_trade.entry_price - fill_price) * open_trade.amount
                    )
                else:
                    session.add(
                        Trade(
                            symbol=event.symbol,
                            status="OPEN",
                            side="LONG" if event.side.lower() == "buy" else "SHORT",
                            entry_price=fill_price,
                            amount=event.amount,
                            sl_price=event.sl_price,
                            tp_price=event.tp_price,
                            mode=self.settings.TRADING_MODE,
                            order_id=order_id,
                        )
                    )
                await session.commit()

            if not event.is_closing:
                try:
                    stop_orders = await self.exchange.create_stop_orders(
                        symbol=event.symbol,
                        position_side=event.side,
                        amount=event.amount,
                        sl_price=event.sl_price,
                        tp_price=event.tp_price,
                    )
                    if stop_orders is not None and (
                        (event.sl_price and stop_orders.get("sl_order") is None)
                        or (event.tp_price and stop_orders.get("tp_order") is None)
                    ):
                        raise RuntimeError(
                            f"Protective Algo orders were not created for {event.symbol}"
                        )
                except Exception as exc:
                    logger.exception(f"[ExecutionService] Failed to create stop orders for {event.symbol}")
                    await self.bus.publish(
                        ErrorEvent(
                            source="ExecutionService",
                            exception=exc,
                            context=(
                                f"Stop orders were not created for {event.symbol}; "
                                f"trade {order_id} is recorded in the database."
                            ),
                        )
                    )
                    return

            logger.info(f"[ExecutionService] Исполнено: {event.symbol} {event.side} {event.amount} @ ${fill_price}")
            await self.bus.publish(
                OrderExecutedEvent(
                    symbol=event.symbol,
                    side=event.side,
                    amount=event.amount,
                    price=fill_price,
                    order_id=order_id,
                    sl_price=event.sl_price,
                    tp_price=event.tp_price,
                    mode=self.settings.TRADING_MODE,
                )
            )
            if event.is_closing:
                await self.bus.publish(
                    TradeClosedEvent(
                        symbol=event.symbol,
                        side=open_trade.side,
                        amount=open_trade.amount,
                        entry_price=open_trade.entry_price,
                        exit_price=fill_price,
                        pnl=open_trade.pnl,
                        reason="Opposite signal",
                    )
                )
        except Exception as exc:
            logger.exception(f"[ExecutionService] Ошибка исполнения {event.symbol}: {exc}")

    async def monitor_open_trades(self):
        """Проверяет статусы активных позиций."""
        async with AsyncSessionFactory() as session:
            trades = list(
                (await session.execute(
                    select(Trade).where(Trade.status == "OPEN")
                )).scalars()
            )

            for trade in trades:
                pos = await self.exchange.get_position(trade.symbol)
                if pos is None or float(pos.get("amount", 0.0)) == 0.0:
                    trade.status = "CLOSED"
                    trade.closed_at = datetime.now(timezone.utc)
                    trade.exit_price = await self._get_exit_price(trade)
                    pnl = (
                        (trade.exit_price - trade.entry_price) * trade.amount
                        if trade.side == "LONG"
                        else (trade.entry_price - trade.exit_price) * trade.amount
                    )
                    trade.pnl = pnl
                    await session.commit()

                    await self.bus.publish(
                        TradeClosedEvent(
                            symbol=trade.symbol,
                            side=trade.side,
                            amount=trade.amount,
                            entry_price=trade.entry_price,
                            exit_price=trade.exit_price,
                            pnl=pnl,
                            reason="Сработка SL/TP на бирже",
                        )
                    )
                elif isinstance(self.exchange, PaperExchange):
                    await self._check_paper_sl_tp(session, trade)

    async def _check_paper_sl_tp(self, session, trade: Trade):
        """Проверяет касание SL/TP по свечам 1m для Paper-режима."""
        klines = await self.exchange.get_klines(trade.symbol, "1m", limit=1)
        if klines.empty:
            return

        latest = klines.iloc[-1]
        high = float(latest["high"])
        low = float(latest["low"])

        exit_price = None
        reason = None

        if trade.side == "LONG":
            if trade.sl_price and low <= trade.sl_price:
                exit_price = trade.sl_price
                reason = "Сработка Stop-Loss (Paper)"
            elif trade.tp_price and high >= trade.tp_price:
                exit_price = trade.tp_price
                reason = "Сработка Take-Profit (Paper)"
        elif trade.side == "SHORT":
            if trade.sl_price and high >= trade.sl_price:
                exit_price = trade.sl_price
                reason = "Сработка Stop-Loss (Paper)"
            elif trade.tp_price and low <= trade.tp_price:
                exit_price = trade.tp_price
                reason = "Сработка Take-Profit (Paper)"

        if exit_price is not None:
            close_side = "sell" if trade.side == "LONG" else "buy"
            await self.exchange.create_order(
                symbol=trade.symbol,
                side=close_side,
                order_type="market",
                amount=trade.amount,
                price=exit_price,
                reduce_only=True,
            )

            trade.status = "CLOSED"
            trade.closed_at = datetime.now(timezone.utc)
            trade.exit_price = exit_price
            pnl = (
                (exit_price - trade.entry_price) * trade.amount
                if trade.side == "LONG"
                else (trade.entry_price - exit_price) * trade.amount
            )
            trade.pnl = pnl
            await session.commit()

            logger.info(
                f"[ExecutionService] [Paper SL/TP] {trade.symbol} {trade.side} "
                f"закрыт по ${exit_price:.2f} ({reason}), PnL: ${pnl:+.2f}"
            )
            await self.bus.publish(
                TradeClosedEvent(
                    symbol=trade.symbol,
                    side=trade.side,
                    amount=trade.amount,
                    entry_price=trade.entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    reason=reason,
                )
            )

    async def _get_exit_price(self, trade: Trade) -> float:
        klines = await self.exchange.get_klines(trade.symbol, "1m", limit=1)
        if klines.empty:
            return trade.sl_price or trade.entry_price

        current_price = float(klines.iloc[-1]["close"])
        is_profitable = (
            current_price >= trade.entry_price
            if trade.side == "LONG"
            else current_price <= trade.entry_price
        )
        return (trade.tp_price if is_profitable else trade.sl_price) or trade.entry_price