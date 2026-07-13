import html
import os
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

import src.keyboards.user as kb
from src.services.user import UserService
from src.states.user import FeedbackForm

from src.core.config import get_settings
from src.crud.paper import PaperTradingRepository
from src.crud.kline import KlineRepository
from src.models.predictor import Predictor

router = Router()


@router.message(Command("status"))
async def status_handler(message: Message, session: AsyncSession):
    """
    Команда /status: Выводит текущее состояние виртуального баланса
    и параметры открытой сделки с расчетом PnL в реальном времени.
    """
    repo = PaperTradingRepository(session)
    portfolio = await repo.get_portfolio()
    active_trade = await repo.get_active_trade("BTC/USDT")

    status_text = (
        f"📊 <b>Виртуальный портфель (Paper Trading)</b>\n\n"
        f"💵 Свободный кэш: <code>{portfolio.cash:.2f}$</code>\n"
        f"📈 Общий баланс: <code>{portfolio.balance:.2f}$</code>\n\n"
    )

    if active_trade:
        # Пытаемся взять последнюю свечу из БД для оценки текущей прибыли в реальном времени
        kline_repo = KlineRepository(session)
        klines = await kline_repo.get_klines("BTC/USDT", "1h", limit=1)

        current_price_str = ""
        if klines:
            current_close = klines[0].close
            unrealized_pnl = (
                current_close - active_trade.entry_price
            ) * active_trade.amount
            current_price_str = (
                f"🎯 Текущая цена: <code>{current_close:.2f}$</code>\n"
                f"💰 Текущий PnL: <code>{unrealized_pnl:+.2f}$</code>\n"
            )

        status_text += (
            f"🚀 <b>Активная позиция по {active_trade.symbol}:</b>\n"
            f"📥 Цена входа: <code>{active_trade.entry_price:.2f}$</code>\n"
            f"📦 Объем: <code>{active_trade.amount:.6f} монет</code>\n"
            f"🛑 Stop-Loss: <code>{active_trade.sl_price:.2f}$</code>\n"
            f"🎯 Take-Profit: <code>{active_trade.tp_price:.2f}$</code>\n"
            f"{current_price_str}"
        )
    else:
        status_text += "📭 <i>Активных позиций нет. Бот находится вне рынка.</i>"

    await message.answer(status_text)


@router.message(Command("signals"))
async def signals_handler(message: Message, session: AsyncSession):
    """
    Команда /signals: Ручной принудительный опрос ML-модели по текущим свечам из БД.
    """
    settings = get_settings()

    # 1. Проверяем, обучена ли модель
    if not os.path.exists(settings.MODEL_PATH):
        await message.answer(
            "⚠️ <b>Модель еще не обучена.</b>\n\n"
            "Пожалуйста, сначала соберите датасет и обучите модель (LGBM), "
            "чтобы файл модели сохранился на сервере."
        )
        return

    # 2. Считываем свечи из БД
    kline_repo = KlineRepository(session)
    klines = await kline_repo.get_klines("BTC/USDT", "1h", limit=50)

    if len(klines) < 30:
        await message.answer(
            f"⚠️ <b>Недостаточно свечей в БД для анализа.</b>\n\n"
            f"Имеется: {len(klines)} свечей. Требуется минимум: 30."
        )
        return

    # Превращаем свечи в DataFrame для предсказателя
    data = []
    for k in klines:
        data.append(
            {
                "open_time": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
        )
    df = pd.DataFrame(data).sort_values("open_time").reset_index(drop=True)

    # 3. Загружаем модель и вычисляем сигнал
    try:
        predictor = Predictor(settings.MODEL_PATH)
        prediction = predictor.predict(df)

        if prediction is None:
            await message.answer(
                "⚠️ Ошибка: не удалось рассчитать признаки для прогноза."
            )
            return

        if prediction == 1:
            recommendation = "🟢 <b>ПОКУПКА (LONG)</b>"
            details = "Модель прогнозирует импульс роста цены в ближайшие часы."
        else:
            recommendation = "🔴 <b>ВНЕ РЫНКА (HOLD / FLAT)</b>"
            details = (
                "Модель не видит сильного восходящего потенциала цены в данный момент."
            )

        await message.answer(
            f"🤖 <b>Анализ рынка от MarketMind</b>\n"
            f"📊 Валютная пара: <code>BTC/USDT</code>\n"
            f"⏱ Таймфрейм: <code>1h</code>\n\n"
            f"🎯 Рекомендация: {recommendation}\n"
            f"📝 Описание: {details}"
        )
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка при анализе рынка: {e}")


# --- Стандартные хендлеры шаблона ---


async def start_handler(message: Message, session: AsyncSession, redis: Redis):
    service = UserService(session, redis)
    user, is_new = await service.register_or_update(
        user_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name,
    )
    greeting = "Привет" if is_new else "С возвращением"
    await message.answer(
        f"{greeting}, {html.escape(message.from_user.full_name)}! 👋\n\n"
        f"<b>Доступные команды торгового ИИ:</b>\n"
        f"/status — Состояние виртуального портфеля\n"
        f"/signals — Получить торговый сигнал ИИ",
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

    # Связываем команду /status и кнопку "📊 Статус портфеля" с одним обработчиком
    router.message.register(status_handler, Command("status"))
    router.message.register(status_handler, F.text == "📊 Статус портфеля")

    # Связываем команду /signals и кнопку "🤖 Торговый сигнал" с одним обработчиком
    router.message.register(signals_handler, Command("signals"))
    router.message.register(signals_handler, F.text == "🤖 Торговый сигнал")

    router.message.register(menu_handler, Command("menu"))
    router.message.register(menu_handler, F.text == "📋 Меню")
    router.message.register(feedback_start, F.text == "💬 Обратная связь")
    router.message.register(
        feedback_text_received, FeedbackForm.waiting_for_text, F.text
    )
    router.message.register(feedback_cancel, F.text == "/cancel")
