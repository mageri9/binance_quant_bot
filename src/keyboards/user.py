from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder


def main_menu() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    # Верхний ряд — основные функции количественного анализа
    builder.row(
        KeyboardButton(text="📊 Статус портфеля"),
        KeyboardButton(text="🤖 Торговый сигнал"),
    )
    # Нижний ряд — меню шаблона и связь
    builder.row(
        KeyboardButton(text="📋 Меню"),
        KeyboardButton(text="💬 Обратная связь"),
    )
    return builder.as_markup(resize_keyboard=True)


def sub_menu() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="ℹ️ О боте", callback_data="about"),
        InlineKeyboardButton(text="❌ Отмена", callback_data="cancel"),
    )
    return builder.as_markup()


def confirm_keyboard(action: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Да", callback_data=f"confirm:{action}"),
        InlineKeyboardButton(text="❌ Нет", callback_data="cancel"),
    )
    return builder.as_markup()


def rating_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(*[
        InlineKeyboardButton(text=str(i), callback_data=f"rating:{i}")
        for i in range(1, 6)
    ])
    builder.row(InlineKeyboardButton(text="❌ Отмена", callback_data="cancel"))
    return builder.as_markup()
