import html

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

import src.keyboards.user as kb
from src.services.user import UserService
from src.states.user import FeedbackForm

router = Router()


async def start_handler(message: Message, session: AsyncSession, redis: Redis):
    service = UserService(session, redis)
    user, is_new = await service.register_or_update(
        user_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name,
    )
    greeting = "Привет" if is_new else "С возвращением"
    await message.answer(
        f"{greeting}, {html.escape(message.from_user.full_name)}! 👋",
        reply_markup=kb.main_menu(),
    )


async def menu_handler(message: Message):
    await message.answer("📋 Меню", reply_markup=kb.sub_menu())


# --- FSM: Feedback flow ---

async def feedback_start(message: Message, state: FSMContext):
    await state.set_state(FeedbackForm.waiting_for_text)
    await message.answer("✍️ Напишите ваш отзыв:")


async def feedback_text_received(message: Message, state: FSMContext):
    await state.update_data(text=message.text)
    await state.set_state(FeedbackForm.waiting_for_rating)
    await message.answer("⭐ Оцените нас от 1 до 5:", reply_markup=kb.rating_keyboard())


async def feedback_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("❌ Отменено.", reply_markup=kb.main_menu())


def register_handlers():
    router.message.register(start_handler, CommandStart())
    router.message.register(menu_handler, Command("menu"))
    router.message.register(menu_handler, F.text == "📋 Меню")
    router.message.register(feedback_start, F.text == "💬 Обратная связь")
    router.message.register(
        feedback_text_received, FeedbackForm.waiting_for_text, F.text
    )
    router.message.register(
        feedback_cancel, F.text == "/cancel"
    )
