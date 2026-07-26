from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def main_keyboard() -> ReplyKeyboardMarkup:
    """Главная клавиатура бота."""
    kb = [
        [KeyboardButton(text="/status"), KeyboardButton(text="/positions")],
        [KeyboardButton(text="/trades"), KeyboardButton(text="/risk")],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)