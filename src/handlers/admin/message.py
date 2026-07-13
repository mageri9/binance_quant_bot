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
    router.message.register(
        broadcast_text_handler, BroadcastStates.waiting_for_text, IsAdmin()
    )
