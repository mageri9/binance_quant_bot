import pytest
from unittest.mock import AsyncMock, patch
from aiogram.types import Message, Chat, User

from src.handlers.user.message import (
    status_handler,
    signals_handler,
    report_handler,
    subscribe_handler,
    unsubscribe_handler
)
import html

from aiogram import Router
from aiogram.filters import Command, StateFilter
from aiogram.types import Message
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from sqlalchemy.ext.asyncio import AsyncSession
from aiogram.exceptions import TelegramAPIError, TelegramForbiddenError

from src.crud.user import UserRepository
from src.filters.check_admin import IsAdmin

# Автоматическая изоляция тестов обработчиков от реальных сетевых вызовов Binance
@pytest.fixture(autouse=True)
def mock_binance_exchange():
    with patch("src.exchange.binance.BinanceExchange") as mock_class:
        instance = mock_class.return_value
        instance.get_balance = AsyncMock(return_value={"free": 500.0, "total": 1500.0})
        instance.close = AsyncMock()
        yield instance

router = Router()


# Определяем состояния для машины состояний (FSM)
class BroadcastStates(StatesGroup):
    waiting_for_text = State()


async def admin_handler(message: Message):
    """Приветственное сообщение админ-панели."""
    await message.answer(
        f"👑 Привет, {html.escape(message.from_user.full_name)}!\n\n"
        f"Команды:\n"
        f"/stats — статистика пользователей\n"
        f"/broadcast — рассылка сообщений всем пользователям\n"
        f"/cancel — отмена текущего действия"
    )


async def stats_handler(message: Message, session: AsyncSession):
    """Вывод количества активных пользователей."""
    repo = UserRepository(session)
    users = await repo.get_all_active()
    await message.answer(f"📊 Активных пользователей в БД: {len(users)}")


async def broadcast_start_handler(message: Message, state: FSMContext):
    """Запуск процесса рассылки. Переводит бота в режим ожидания текста."""
    await message.answer(
        "📝 <b>Режим рассылки сообщений</b>\n\n"
        "Отправьте текст, который вы хотите разослать ВСЕМ активным пользователям бота.\n"
        "Вы можете использовать HTML-разметку.\n\n"
        "Для отмены операции введите команду /cancel"
    )
    await state.set_state(BroadcastStates.waiting_for_text)


async def broadcast_cancel_handler(message: Message, state: FSMContext):
    """Сброс состояния и отмена рассылки."""
    await state.clear()
    await message.answer("❌ Рассылка отменена.")


async def broadcast_text_handler(
    message: Message, state: FSMContext, session: AsyncSession
):
    """Прием текста рассылки и осуществление вещания с обработкой ошибок."""
    text_to_send = message.text or message.caption
    if not text_to_send:
        await message.answer("⚠️ Пожалуйста, отправьте текстовое сообщение.")
        return

    # Очищаем состояние FSM, так как текст успешно получен
    await state.clear()
    await message.answer("⏳ Начинаю рассылку сообщений...")

    repo = UserRepository(session)
    active_users = await repo.get_all_active()

    success_count = 0
    blocked_count = 0
    failed_count = 0

    for user in active_users:
        try:
            await message.bot.send_message(chat_id=user.user_id, text=text_to_send)
            success_count += 1
        except TelegramForbiddenError:
            # Пользователь заблокировал бота — помечаем его в БД
            await repo.set_blocked(user.user_id, True)
            blocked_count += 1
        except TelegramAPIError as e:
            err_msg = str(e).lower()
            # Дополнительные проверки на удаленные чаты и деактивированных пользователей
            if (
                "chat not found" in err_msg
                or "deactivated" in err_msg
                or "blocked" in err_msg
            ):
                await repo.set_blocked(user.user_id, True)
                blocked_count += 1
            else:
                failed_count += 1
        except Exception:
            failed_count += 1

    await message.answer(
        f"✅ <b>Рассылка успешно завершена!</b>\n\n"
        f"📥 Доставлено: <code>{success_count}</code>\n"
        f"🚫 Заблокировали бота (и отключены в БД): <code>{blocked_count}</code>\n"
        f"❌ Ошибок сети/отправки: <code>{failed_count}</code>"
    )


def register_handlers():
    router.message.register(admin_handler, Command("admin"), IsAdmin())
    router.message.register(stats_handler, Command("stats"), IsAdmin())

    # Обработчики рассылки
    router.message.register(
        broadcast_cancel_handler, Command("cancel"), IsAdmin(), StateFilter("*")
    )
    router.message.register(broadcast_start_handler, Command("broadcast"), IsAdmin())
    router.message.register(broadcast_text_handler, BroadcastStates.waiting_for_text, IsAdmin())



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


@pytest.mark.asyncio
async def test_user_subscribe_handler(temp_db_session):
    from src.crud.user import UserRepository

    repo = UserRepository(temp_db_session)
    # Создаем пользователя (по умолчанию он подписан)
    await repo.create(user_id=444, username="test_sub", full_name="Sub User")

    chat = Chat(id=444, type="private")
    user = User(id=444, is_bot=False, first_name="Sub User")
    message = AsyncMock(spec=Message)
    message.chat = chat
    message.from_user = user
    message.answer = AsyncMock()

    # Сначала отписываемся
    await unsubscribe_handler(message, temp_db_session)

    # Проверяем, что первый переданный аргумент (текст сообщения) содержит нужную фразу
    assert "Вы отписались от уведомлений" in message.answer.call_args[0][0]

    u_record = await repo.get_by_user_id(444)
    assert u_record.is_subscribed is False

    # Снова подписываемся
    await subscribe_handler(message, temp_db_session)

    # Проверяем текст сообщения об успешной подписке
    assert "Вы успешно подписались на уведомления" in message.answer.call_args[0][0]

    u_record = await repo.get_by_user_id(444)
    assert u_record.is_subscribed is True

@pytest.mark.asyncio
async def test_status_handler_shows_short_from_db_field(temp_db_session):
    """
    Регрессионный тест: направление позиции в /status должно браться
    из PaperTrade.is_short, а не пересчитываться эвристикой по
    sl_price/tp_price. Намеренно задаём sl_price/tp_price так, чтобы
    старая эвристика (sl_price > entry_price) ошиблась бы и показала LONG,
    хотя сделка на самом деле SHORT.
    """
    from src.crud.paper import PaperTradingRepository

    repo = PaperTradingRepository(temp_db_session)
    # entry=100, sl_price ниже entry (как у LONG), но is_short=True явно
    await repo.create_trade(
        symbol="BTC/USDT",
        entry_price=100.0,
        amount=1.0,
        sl_price=98.0,   # < entry_price -> старая эвристика решила бы, что это LONG
        tp_price=104.0,  # > entry_price -> тоже указывало бы на LONG по старой логике
        entry_candle_time=1000,
        is_short=True,   # но по факту сделка SHORT
    )

    chat = Chat(id=123, type="private")
    user = User(id=123, is_bot=False, first_name="TestUser")
    message = AsyncMock(spec=Message)
    message.chat = chat
    message.from_user = user
    message.answer = AsyncMock()

    await status_handler(message, temp_db_session)

    message.answer.assert_called_once()
    answer_text = message.answer.call_args[0][0]

    assert "SHORT" in answer_text
    assert "LONG 🟢" not in answer_text


@pytest.mark.asyncio
async def test_status_handler_shows_short_without_klines(temp_db_session):
    """
    То же самое, но для ветки без свежих свечей в БД (klines пуст) —
    там раньше была отдельная эвристика в блоке else.
    """
    from src.crud.paper import PaperTradingRepository

    repo = PaperTradingRepository(temp_db_session)
    await repo.create_trade(
        symbol="BTC/USDT",
        entry_price=100.0,
        amount=1.0,
        sl_price=98.0,
        tp_price=104.0,
        entry_candle_time=1000,
        is_short=True,
    )
    # Никаких klines в БД не сохраняем -> get_klines() вернёт пустой список

    chat = Chat(id=123, type="private")
    user = User(id=123, is_bot=False, first_name="TestUser")
    message = AsyncMock(spec=Message)
    message.chat = chat
    message.from_user = user
    message.answer = AsyncMock()

    await status_handler(message, temp_db_session)

    message.answer.assert_called_once()
    answer_text = message.answer.call_args[0][0]

    assert "SHORT" in answer_text