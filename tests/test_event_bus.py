import pytest
from dataclasses import dataclass
from src.event_bus import AsyncEventBus, Event, ErrorEvent


@dataclass(kw_only=True)
class CustomTestEvent(Event):
    msg: str


@pytest.mark.asyncio
async def test_event_bus_publish_subscribe():
    bus = AsyncEventBus()
    received = []

    async def handle_custom(event: CustomTestEvent):
        received.append(event.msg)

    bus.subscribe(CustomTestEvent, handle_custom)
    await bus.publish(CustomTestEvent(msg="hello"))

    assert len(received) == 1
    assert received[0] == "hello"


@pytest.mark.asyncio
async def test_event_bus_handler_error_publishes_error_event():
    bus = AsyncEventBus()
    errors = []

    async def failing_handler(event: CustomTestEvent):
        raise ValueError("Test Error")

    async def error_listener(event: ErrorEvent):
        errors.append(event)

    bus.subscribe(CustomTestEvent, failing_handler)
    bus.subscribe(ErrorEvent, error_listener)

    await bus.publish(CustomTestEvent(msg="boom"))

    assert len(errors) == 1
    assert errors[0].source == "failing_handler"
    assert isinstance(errors[0].exception, ValueError)