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
from src.crud.user import UserRepository
from src.models.predictor import Predictor
from src.strategy.signals import calculate_strategy_metrics

router = Router()


@router.message(Command("status"))
@router.message(F.text == "📊 Статус портфеля")
async def status_handler(message: Message, session: AsyncSession):
    """Выводит сводное текущее состояние виртуального кошелька по всем парам."""
    repo = PaperTradingRepository(session)
    portfolio = await repo.get_portfolio()
    settings = get_settings()

    active_trades_text = ""

    for symbol, timeframe in settings.ACTIVE_CONFIGS:
        active_trade = await repo.get_active_trade(symbol)

        if active_trade:
            kline_repo = KlineRepository(session)
            klines = await kline_repo.get_klines(symbol, timeframe, limit=1)

            current_price_str = ""
            if klines:
                current_close = klines[0].close

                # Определяем направление сделки LONG или SHORT
                is_short = False
                if active_trade.sl_price is not None:
                    is_short = active_trade.sl_price > active_trade.entry_price
                elif active_trade.tp_price is not None:
                    is_short = active_trade.tp_price < active_trade.entry_price

                if is_short:
                    unrealized_pnl = (
                        active_trade.entry_price - current_close
                    ) * active_trade.amount
                    pos_type = "SHORT 🔴"
                else:
                    unrealized_pnl = (
                        current_close - active_trade.entry_price
                    ) * active_trade.amount
                    pos_type = "LONG 🟢"

                current_price_str = (
                    f"🎯 Текущая цена: <code>{current_close:.2f}$</code>\n"
                    f"💰 Текущий PnL: <code>{unrealized_pnl:+.2f}$</code>\n"
                )
            else:
                pos_type = (
                    "SHORT"
                    if (
                        active_trade.sl_price
                        and active_trade.sl_price > active_trade.entry_price
                    )
                    else "LONG"
                )

            active_trades_text += (
                f"🚀 <b>Активная позиция {pos_type} по {symbol}:</b>\n"
                f"📥 Цена входа: <code>{active_trade.entry_price:.2f}$</code>\n"
                f"📦 Объем: <code>{active_trade.amount:.6f} монет</code>\n"
                f"🛑 Stop-Loss: <code>{active_trade.sl_price:.2f}$</code>\n"
                f"🎯 Take-Profit: <code>{active_trade.tp_price:.2f}$</code>\n"
                f"{current_price_str}\n"
            )
        else:
            active_trades_text += (
                f"📭 <b>{symbol}:</b> <i>Вне рынка. Бот ожидает сигнала.</i>\n\n"
            )

    status_text = (
        f"📊 <b>Виртуальный портфель (Multi-Asset Paper Trading)</b>\n\n"
        f"💵 Свободный кэш: <code>{portfolio.cash:.2f}$</code>\n"
        f"📈 Общий баланс: <code>{portfolio.balance:.2f}$</code>\n\n"
        f"{active_trades_text}"
    )

    await message.answer(status_text)


@router.message(Command("signals"))
@router.message(F.text == "🤖 Торговый сигнал")
async def signals_handler(message: Message, session: AsyncSession):
    """Ручной опрос всех активных моделей по текущим ценам в БД."""
    settings = get_settings()
    signals_text = "🤖 <b>Анализ рынка от MarketMind</b>\n\n"

    for symbol, timeframe in settings.ACTIVE_CONFIGS:
        model_path = settings.get_model_path(symbol, timeframe)

        if not os.path.exists(model_path):
            signals_text += (
                f"⚠️ <b>{symbol} ({timeframe}):</b> Модель еще не обучена.\n\n"
            )
            continue

        kline_repo = KlineRepository(session)
        klines = await kline_repo.get_klines(symbol, timeframe, limit=50)

        if len(klines) < 30:
            signals_text += f"⚠️ <b>{symbol} ({timeframe}):</b> Недостаточно свечей ({len(klines)}/30).\n\n"
            continue

        data = [
            {
                "open_time": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
            for k in klines
        ]
        df = pd.DataFrame(data).sort_values("open_time").reset_index(drop=True)

        try:
            predictor = Predictor(model_path)
            prediction = predictor.predict(df)

            if prediction == 1:
                recommendation = "🟢 <b>ПОКУПКА (LONG)</b>"
                details = "Прогнозируется рост цены."
            elif prediction == -1:
                recommendation = "🔴 <b>ПРОДАЖА (SHORT)</b>"
                details = "Прогнозируется падение цены."
            else:
                recommendation = "⚪️ <b>ВНЕ РЫНКА (HOLD)</b>"
                details = "Сильных трендовых импульсов не обнаружено."

            signals_text += (
                f"📊 <b>{symbol} ({timeframe}):</b>\n"
                f"🎯 Рекомендация: {recommendation}\n"
                f"📝 {details}\n\n"
            )
        except Exception as e:
            signals_text += f"❌ <b>{symbol} ({timeframe}):</b> Ошибка анализа: {e}\n\n"

    await message.answer(signals_text)


@router.message(Command("report"))
@router.message(F.text == "📈 Отчёт по стратегии")
async def report_handler(message: Message, session: AsyncSession):
    """Выводит сводный отчет по всем закрытым сделкам портфеля."""
    settings = get_settings()
    repo = PaperTradingRepository(session)

    all_closed_trades = []
    for symbol, timeframe in settings.ACTIVE_CONFIGS:
        closed_trades = await repo.get_closed_trades(symbol)
        all_closed_trades.extend(closed_trades)

    if not all_closed_trades:
        await message.answer(
            "📭 <i>Пока нет ни одной закрытой сделки. Отчёт появится после первых результатов.</i>"
        )
        return

    # Сортируем все сделки по хронологии входа для правильного расчета просадок
    all_closed_trades.sort(key=lambda t: t.entry_candle_time)

    trade_returns = []
    for t in all_closed_trades:
        # Для шортов и лонгов расчет доходности отличается
        is_short = False
        if t.sl_price is not None:
            is_short = t.sl_price > t.entry_price
        elif t.tp_price is not None:
            is_short = t.tp_price < t.entry_price

        if is_short:
            ret = (t.entry_price - t.exit_price) / t.entry_price
        else:
            ret = (t.exit_price - t.entry_price) / t.entry_price

        trade_returns.append(ret)

    metrics = calculate_strategy_metrics(trade_returns)

    await message.answer(
        f"📈 <b>Сводный отчёт по стратегии (Multi-Asset)</b>\n\n"
        f"🔢 Всего сделок (суммарно): <code>{metrics['total_trades']}</code>\n"
        f"✅ Win rate системы: <code>{metrics['win_rate']:.1%}</code>\n"
        f"💹 Profit Factor: <code>{metrics['profit_factor']:.2f}</code>\n"
        f"📊 Общий Sharpe: <code>{metrics['sharpe_ratio']:.3f}</code>\n"
        f"📊 Общий Sortino: <code>{metrics['sortino_ratio']:.3f}</code>\n"
        f"📉 Макс. просадка портфеля: <code>{metrics['max_drawdown']:.1%}</code>\n"
        f"🎯 Матожидание (Expectancy): <code>{metrics['expectancy']:.3%}</code> на сделку\n"
        f"💰 Накопленная доходность: <code>{metrics['total_return']:.1%}</code>"
    )


@router.message(Command("subscribe"))
@router.message(F.text == "🔔 Подписаться на сигналы")
async def subscribe_handler(message: Message, session: AsyncSession):
    repo = UserRepository(session)
    await repo.set_subscribed(message.from_user.id, True)
    await message.answer(
        "🔔 <b>Вы успешно подписались на уведомления о сделках!</b>\n\n"
        "Теперь вы будете получать сообщения о закрытии позиций в реальном времени.",
        reply_markup=kb.main_menu(is_subscribed=True),
    )


@router.message(Command("unsubscribe"))
@router.message(F.text == "🔕 Отписаться от сигналов")
async def unsubscribe_handler(message: Message, session: AsyncSession):
    repo = UserRepository(session)
    await repo.set_subscribed(message.from_user.id, False)
    await message.answer(
        "🔕 <b>Вы отписались от уведомлений о сделках.</b>\n\n"
        "Вы больше не будете получать сообщения о закрытых позициях.",
        reply_markup=kb.main_menu(is_subscribed=False),
    )


async def start_handler(message: Message, session: AsyncSession, redis: Redis):
    service = UserService(session, redis)
    user, is_new = await service.register_or_update(
        user_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name,
    )
    greeting = "Привет" if is_new else "С возвращением"

    is_sub = getattr(user, "is_subscribed", True)

    await message.answer(
        f"{greeting}, {html.escape(message.from_user.full_name)}! 👋\n\n"
        f"<b>Доступные функции количественного ИИ:</b>\n"
        f"👉 Нажмите на кнопки внизу для взаимодействия.",
        reply_markup=kb.main_menu(is_subscribed=is_sub),
    )


def register_handlers():
    router.message.register(start_handler, CommandStart())
    router.message.register(subscribe_handler, Command("subscribe"))
    router.message.register(subscribe_handler, F.text == "🔔 Подписаться на сигналы")
    router.message.register(unsubscribe_handler, Command("unsubscribe"))
    router.message.register(unsubscribe_handler, F.text == "🔕 Отписаться от сигналов")
    router.message.register(status_handler, Command("status"))
    router.message.register(status_handler, F.text == "📊 Статус портфеля")
    router.message.register(signals_handler, Command("signals"))
    router.message.register(signals_handler, F.text == "🤖 Торговый сигнал")
    router.message.register(report_handler, Command("report"))
    router.message.register(report_handler, F.text == "📈 Отчёт по стратегии")
