import pytest
from unittest.mock import AsyncMock, patch
from aiogram.types import Message, Chat, User

from src.handlers.user.message import status_handler, signals_handler, report_handler



@pytest.mark.asyncio
async def test_status_handler_no_trades(temp_db_session):
    # Создаем фиктивное сообщение от пользователя
    chat = Chat(id=123, type="private")
    user = User(id=123, is_bot=False, first_name="TestUser")
    message = AsyncMock(spec=Message)
    message.chat = chat
    message.from_user = user
    message.answer = AsyncMock()

    # Запускаем команду /status
    await status_handler(message, temp_db_session)

    # Проверяем, что бот прислал красивый ответ пользователю
    message.answer.assert_called_once()
    answer_text = message.answer.call_args[0][0]

    assert "Виртуальный портфель" in answer_text
    assert "Активных позиций нет" in answer_text


@pytest.mark.asyncio
async def test_signals_handler_no_model(temp_db_session):
    chat = Chat(id=123, type="private")
    user = User(id=123, is_bot=False, first_name="TestUser")
    message = AsyncMock(spec=Message)
    message.chat = chat
    message.from_user = user
    message.answer = AsyncMock()

    # Запускаем команду /signals при отсутствии обученной модели на диске
    await signals_handler(message, temp_db_session)

    # Бот должен предупредить, что модель еще не обучена
    message.answer.assert_called_once()
    answer_text = message.answer.call_args[0][0]

    assert "Модель еще не обучена" in answer_text

@pytest.mark.asyncio
async def test_report_handler_no_trades(temp_db_session):
    chat = Chat(id=123, type="private")
    user = User(id=123, is_bot=False, first_name="TestUser")
    message = AsyncMock(spec=Message)
    message.chat = chat
    message.from_user = user
    message.answer = AsyncMock()

    await report_handler(message, temp_db_session)

    message.answer.assert_called_once()
    assert "Пока нет ни одной закрытой сделки" in message.answer.call_args[0][0]


@pytest.mark.asyncio
async def test_report_handler_with_trades(temp_db_session):
    from src.crud.paper import PaperTradingRepository

    repo = PaperTradingRepository(temp_db_session)
    trade = await repo.create_trade(
        symbol="BTC/USDT", entry_price=100.0, amount=1.0,
        sl_price=98.0, tp_price=104.0, entry_candle_time=1000,
    )
    await repo.close_trade(trade, exit_price=104.0, pnl=4.0)

    chat = Chat(id=123, type="private")
    user = User(id=123, is_bot=False, first_name="TestUser")
    message = AsyncMock(spec=Message)
    message.chat = chat
    message.from_user = user
    message.answer = AsyncMock()

    await report_handler(message, temp_db_session)

    message.answer.assert_called_once()
    answer_text = message.answer.call_args[0][0]
    assert "Отчёт по стратегии" in answer_text
    assert "Win rate" in answer_text