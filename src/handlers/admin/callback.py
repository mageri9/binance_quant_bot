from aiogram import F, Router
from aiogram.types import CallbackQuery

from src.filters.check_admin import IsAdmin

router = Router()


async def admin_cancel_handler(query: CallbackQuery):
    await query.message.edit_text("❌ Операция отменена.")


def register_handlers():
    router.callback_query.register(admin_cancel_handler, F.data == "admin:cancel", IsAdmin())
