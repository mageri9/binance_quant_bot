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
    builder.row(
        KeyboardButton(text="📊 Статус портфеля"),
        KeyboardButton(text="🤖 Торговый сигнал"),
    )
    builder.row(
        KeyboardButton(text="📈 Отчёт по стратегии"),
    )

    # Добавляем интерактивную кнопку переключения подписки
    if is_subscribed:
        builder.row(KeyboardButton(text="🔕 Отписаться от сигналов"))
    else:
        builder.row(KeyboardButton(text="🔔 Подписаться на сигналы"))

    return builder.as_markup(resize_keyboard=True)
