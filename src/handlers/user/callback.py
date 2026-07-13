from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from src.states.user import FeedbackForm
import src.keyboards.user as kb

router = Router()


async def cancel_handler(query: CallbackQuery, state: FSMContext):
    await state.clear()
    await query.message.edit_text("❌ Операция отменена.")


async def about_handler(query: CallbackQuery):
    await query.answer()
    await query.message.edit_text(
        "ℹ️ Это шаблонный бот на Aiogram 3.\n\nСтек: SQLite + SQLAlchemy + Redis + Docker",
        reply_markup=kb.sub_menu(),
    )


async def rating_handler(query: CallbackQuery, state: FSMContext):
    rating = query.data.split(":")[1]
    data = await state.get_data()
    await state.clear()

    feedback_text = data.get("text", "—")
    await query.message.edit_text(
        f"✅ Спасибо за отзыв!\n\n"
        f"📝 Текст: {feedback_text}\n"
        f"⭐ Оценка: {rating}/5"
    )


def register_handlers():
    router.callback_query.register(cancel_handler, F.data == "cancel")
    router.callback_query.register(about_handler, F.data == "about")
    router.callback_query.register(
        rating_handler, FeedbackForm.waiting_for_rating, F.data.startswith("rating:")
    )
