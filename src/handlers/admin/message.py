import html
import asyncio

from aiogram import Router
from aiogram.filters import Command, StateFilter
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from sqlalchemy.ext.asyncio import AsyncSession
from aiogram.exceptions import TelegramAPIError, TelegramForbiddenError
from aiogram.utils.keyboard import InlineKeyboardBuilder
from redis.asyncio import Redis

from src.crud.user import UserRepository
from src.filters.check_admin import IsAdmin
from src.risk import KillSwitchManager
import src.keyboards.user as user_kb

router = Router()


class BroadcastStates(StatesGroup):
    waiting_for_text = State()


def get_risk_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(
            text="🔄 Сбросить и сверить", callback_data="admin:risk:reset"
        ),
    )
    builder.row(
        InlineKeyboardButton(
            text="🚨 Экстренный Стоп", callback_data="admin:risk:kill"
        ),
        InlineKeyboardButton(
            text="🟢 Активировать NORMAL", callback_data="admin:risk:normal"
        ),
    )
    return builder.as_markup()


async def admin_handler(message: Message):
    admin_keyboard = user_kb.admin_menu()
    await message.answer(
        f"👑 Привет, {html.escape(message.from_user.full_name)}!\n\n"
        f"Команды:\n"
        f"/stats — статистика пользователей\n"
        f"/risk — панель безопасности (SRE Kill Switch)\n"
        f"/broadcast — рассылка сообщений всем пользователям\n"
        f"/cancel — отмена текущего действия"
        ,
        reply_markup=admin_keyboard,
    )


async def stats_handler(message: Message, session: AsyncSession):
    repo = UserRepository(session)
    users = await repo.get_all_active()
    await message.answer(f"📊 Активных пользователей в БД: {len(users)}")


async def risk_handler(message: Message, redis: Redis):
    """Показывает текущее состояние Kill Switch и панель управления"""
    manager = KillSwitchManager(redis)
    state, reason, details = await manager.get_state()

    color = "🟢" if state == "NORMAL" else "🟡" if state == "SAFE_MODE" else "🔴"

    text = (
        f"🛡️ <b>Управление рисками (SRE Kill Switch)</b>\n\n"
        f"📊 Текущее состояние: {color} <b>{state}</b>\n"
    )
    if state != "NORMAL":
        text += (
            f"❓ Причина останова: <code>{reason or 'не указана'}</code>\n"
            f"📝 Детали расхождений:\n<code>{details or 'нет деталей'}</code>\n"
        )
    else:
        text += "✅ Система работает в штатном режиме, блокировок нет.\n"

    await message.answer(text, reply_markup=get_risk_keyboard())


async def broadcast_start_handler(message: Message, state: FSMContext):
    await message.answer(
        "📝 <b>Режим рассылки сообщений</b>\n\n"
        "Отправьте текст, который вы хотите разослать ВСЕМ активным пользователям бота.\n"
        "Вы можете использовать HTML-разметку.\n\n"
        "Для отмены операции введите команду /cancel"
    )
    await state.set_state(BroadcastStates.waiting_for_text)


async def broadcast_cancel_handler(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("❌ Рассылка отменена.")


async def broadcast_text_handler(
    message: Message, state: FSMContext, session: AsyncSession
):
    text_to_send = message.text or message.caption
    if not text_to_send:
        await message.answer("⚠️ Пожалуйста, отправьте текстовое сообщение.")
        return

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
            await repo.set_blocked(user.user_id, True)
            blocked_count += 1
        except TelegramAPIError as e:
            err_msg = str(e).lower()
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
        finally:
            await asyncio.sleep(0.05)

    await message.answer(
        f"✅ <b>Рассылка успешно завершена!</b>\n\n"
        f"📥 Доставлено: <code>{success_count}</code>\n"
        f"🚫 Заблокировали бота (и отключены в БД): <code>{blocked_count}</code>\n"
        f"❌ Ошибок сети/отправки: <code>{failed_count}</code>"
    )


def register_handlers():
    router.message.register(admin_handler, Command("admin"), IsAdmin())
    router.message.register(stats_handler, Command("stats"), IsAdmin())
    router.message.register(risk_handler, Command("risk"), IsAdmin())

    router.message.register(
        broadcast_cancel_handler, Command("cancel"), IsAdmin(), StateFilter("*")
    )
    router.message.register(broadcast_start_handler, Command("broadcast"), IsAdmin())
    router.message.register(
        broadcast_text_handler, BroadcastStates.waiting_for_text, IsAdmin()
    )
