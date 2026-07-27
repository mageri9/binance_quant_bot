import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Type
from loguru import logger


# --- ДАТАКЛАССЫ СОБЫТИЙ (Event Contracts) ---

@dataclass(kw_only=True)
class Event:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(kw_only=True)
class CandleClosedEvent(Event):
    """Событие: закрылась новая свеча на бирже."""
    symbol: str
    timeframe: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(kw_only=True)
class SignalEmittedEvent(Event):
    """Событие: стратегия сгенерировала торговый сигнал."""
    symbol: str
    timeframe: str
    signal: int  # 1 = LONG, -1 = SHORT, 0 = HOLD
    expected_return: float
    close_price: float
    open_time: int


@dataclass(kw_only=True)
class OrderApprovedEvent(Event):
    """Событие: RiskEngine одобрил и рассчитал параметры ордера."""
    symbol: str
    side: str  # "buy" или "sell"
    amount: float
    price: float
    sl_price: float
    tp_price: float
    reason: str
    is_closing: bool = False


@dataclass(kw_only=True)
class OrderExecutedEvent(Event):
    """Событие: ордер успешно исполнен (в Paper, Testnet или Mainnet)."""
    symbol: str
    side: str
    amount: float
    price: float
    order_id: str
    sl_price: float
    tp_price: float
    mode: str


@dataclass(kw_only=True)
class TradeClosedEvent(Event):
    """Событие: позиция закрыта (по TP/SL или сигналу)."""
    symbol: str
    side: str
    amount: float
    entry_price: float
    exit_price: float
    pnl: float
    reason: str


@dataclass(kw_only=True)
class ErrorEvent(Event):
    """Событие: критическая ошибка в одном из сервисов."""
    source: str
    exception: Exception
    context: str = ""


# --- ЛЕГКОВЕСНАЯ IN-MEMORY ШИНА СОБЫТИЙ ---

EventHandler = Callable[[Any], Coroutine[Any, Any, None]]


class AsyncEventBus:
    """Простая асинхронная In-Memory шина событий."""

    def __init__(self):
        self._subscribers: Dict[Type[Event], List[EventHandler]] = {}

    def subscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"[EventBus] Подписчик {handler.__name__} зарегистрирован на {event_type.__name__}")

    async def publish(self, event: Event) -> None:
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            return

        tasks = [self._safe_execute(handler, event) for handler in handlers]
        await asyncio.gather(*tasks)

    async def _safe_execute(self, handler: EventHandler, event: Event) -> None:
        try:
            await handler(event)
        except Exception as exc:
            logger.exception(f"[EventBus] Ошибка в обработчике {handler.__name__} для {type(event).__name__}: {exc}")
            if not isinstance(event, ErrorEvent):
                await self.publish(ErrorEvent(source=handler.__name__, exception=exc, context=str(event)))
