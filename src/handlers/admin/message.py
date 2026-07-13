import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from src.crud.user import UserRepository
from src.filters.check_admin import IsAdmin

router = Router()


async def admin_handler(message: Message):
    await message.answer(
        f"👑 Привет, {html.escape(message.from_user.full_name)}!\n\n"
        f"Команды:\n"
        f"/stats — статистика\n"
        f"/broadcast — рассылка (в разработке)"
    )


async def stats_handler(message: Message, session: AsyncSession):
    repo = UserRepository(session)
    users = await repo.get_all_active()
    await message.answer(f"📊 Активных пользователей: {len(users)}")


def register_handlers():
    router.message.register(admin_handler, Command("admin"), IsAdmin())
    router.message.register(stats_handler, Command("stats"), IsAdmin())
