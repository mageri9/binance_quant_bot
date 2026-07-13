import html
import os
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

import src.keyboards.user as kb
from src.services.user import UserService

from src.core.config import get_settings
from src.crud.paper import PaperTradingRepository
from src.crud.kline import KlineRepository
from src.models.predictor import Predictor
from src.strategy.signals import calculate_strategy_metrics

router = Router()


@router.message(Command("status"))
@router.message(F.text == "📊 Статус портфеля")
async def status_handler(message: Message, session: AsyncSession):
    """
    Команда /status (или кнопка): Выводит текущее состояние виртуального кошелька.
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
@router.message(F.text == "🤖 Торговый сигнал")
async def signals_handler(message: Message, session: AsyncSession):
    """
    Команда /signals (или кнопка): Ручной опрос ML-модели по текущим ценам в БД.
    """
    settings = get_settings()

    if not os.path.exists(settings.MODEL_PATH):
        await message.answer(
            "⚠️ <b>Модель еще не обучена.</b>\n\n"
            "Пожалуйста, соберите датасет и запустите обучение модели (LGBM), "
            "чтобы файл модели сохранился на сервере."
        )
        return

    kline_repo = KlineRepository(session)
    klines = await kline_repo.get_klines("BTC/USDT", "1h", limit=50)

    if len(klines) < 30:
        await message.answer(
            f"⚠️ <b>Недостаточно свечей в БД для анализа.</b>\n\n"
            f"Имеется: {len(klines)} свечей. Требуется минимум: 30."
        )
        return

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


@router.message(Command("report"))
@router.message(F.text == "📈 Отчёт по стратегии")
async def report_handler(message: Message, session: AsyncSession):
    """
    Команда /report: реальные стратегические метрики (Sharpe, Profit Factor,
    Max Drawdown и т.д.) по фактическим закрытым сделкам paper trading —
    не по бэктесту, а по тому, что бот действительно "наторговал".
    """
    repo = PaperTradingRepository(session)
    closed_trades = await repo.get_closed_trades("BTC/USDT")

    if not closed_trades:
        await message.answer(
            "📭 <i>Пока нет ни одной закрытой сделки. Отчёт появится после первых результатов.</i>"
        )
        return

    trade_returns = [
        (t.exit_price - t.entry_price) / t.entry_price for t in closed_trades
    ]
    metrics = calculate_strategy_metrics(trade_returns)

    await message.answer(
        f"📈 <b>Отчёт по стратегии (Paper Trading)</b>\n\n"
        f"🔢 Всего сделок: <code>{metrics['total_trades']}</code>\n"
        f"✅ Win rate: <code>{metrics['win_rate']:.1%}</code>\n"
        f"💹 Profit Factor: <code>{metrics['profit_factor']:.2f}</code>\n"
        f"📊 Sharpe: <code>{metrics['sharpe_ratio']:.3f}</code>\n"
        f"📊 Sortino: <code>{metrics['sortino_ratio']:.3f}</code>\n"
        f"📉 Max Drawdown: <code>{metrics['max_drawdown']:.1%}</code>\n"
        f"🎯 Expectancy: <code>{metrics['expectancy']:.3%}</code> на сделку\n"
        f"💰 Суммарная доходность: <code>{metrics['total_return']:.1%}</code>"
    )


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
        f"<b>Доступные функции количественного ИИ:</b>\n"
        f"👉 Нажмите на кнопки внизу для взаимодействия.",
        reply_markup=kb.main_menu(),
    )


def register_handlers():
    router.message.register(start_handler, CommandStart())
    router.message.register(status_handler, Command("status"))
    router.message.register(status_handler, F.text == "📊 Статус портфеля")
    router.message.register(signals_handler, Command("signals"))
    router.message.register(signals_handler, F.text == "🤖 Торговый сигнал")
    router.message.register(report_handler, Command("report"))
    router.message.register(report_handler, F.text == "📈 Отчёт по стратегии")
