from aiogram.types import (
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def main_menu() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text="📊 Статус портфеля"),
        KeyboardButton(text="🤖 Торговый сигнал"),
    )
    return builder.as_markup(resize_keyboard=True)