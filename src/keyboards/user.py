from aiogram.types import (
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def main_menu(is_subscribed: bool = True) -> ReplyKeyboardMarkup:
    """
    Генерирует главное меню. Кнопка подписки меняется динамически.
    """
    builder = ReplyKeyboardBuilder()
    builder.row(KeyboardButton(text="/portfolio"), KeyboardButton(text="/positions"))
    builder.row(KeyboardButton(text="/trades"), KeyboardButton(text="/health"))
    builder.row(KeyboardButton(text="/models"), KeyboardButton(text="/model ETHUSDT"))
    builder.row(
        KeyboardButton(text="/status"),
        KeyboardButton(text="/signals"),
        KeyboardButton(text="/report"),
    )

    # Добавляем интерактивную кнопку переключения подписки
    if is_subscribed:
        builder.row(KeyboardButton(text="🔕 Отписаться от сигналов"))
    else:
        builder.row(KeyboardButton(text="🔔 Подписаться на сигналы"))

    return builder.as_markup(resize_keyboard=True)


def admin_menu() -> ReplyKeyboardMarkup:
    """Keep elevated operations out of the regular user keyboard."""
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text="/health"),
        KeyboardButton(text="/reconcile"),
        KeyboardButton(text="/risk"),
    )
    builder.row(KeyboardButton(text="/stats"), KeyboardButton(text="/broadcast"))
    return builder.as_markup(resize_keyboard=True)
