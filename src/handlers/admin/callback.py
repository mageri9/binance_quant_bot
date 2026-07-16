from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.filters.check_admin import IsAdmin
from src.risk import KillSwitchManager, KillSwitchState, reconcile_positions
from src.exchange.paper import PaperExchange
from src.exchange.binance import BinanceExchange
from src.core.config import get_settings
from src.crud.paper import PaperTradingRepository

router = Router()


def get_risk_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="🔄 Сбросить и сверить", callback_data="admin:risk:reset"),
    )
    builder.row(
        InlineKeyboardButton(text="🚨 Экстренный Стоп", callback_data="admin:risk:kill"),
        InlineKeyboardButton(text="🟢 Активировать NORMAL", callback_data="admin:risk:normal"),
    )
    builder.row(
        InlineKeyboardButton(text="🔧 Синхронизировать БД под Биржу", callback_data="admin:risk:sync_db"),
    )
    return builder.as_markup()


async def admin_cancel_handler(query: CallbackQuery):
    await query.message.edit_text("❌ Операция отменена.")


async def risk_reset_handler(query: CallbackQuery, session: AsyncSession, redis: Redis):
    await query.answer("Запускаю сверку...")
    manager = KillSwitchManager(redis)
    settings = get_settings()
    symbols = [config[0] for config in settings.ACTIVE_CONFIGS]

    await manager.set_state(KillSwitchState.NORMAL)

    if settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET:
        exchange = BinanceExchange(
            settings.BINANCE_API_KEY,
            settings.BINANCE_API_SECRET,
            settings.BINANCE_TESTNET,
        )
    else:
        exchange = PaperExchange(session)

    try:
        success, error_details = await reconcile_positions(
            exchange, session, symbols, manager
        )
        if success:
            text = (
                "🛡️ <b>Управление рисками (SRE Kill Switch)</b>\n\n"
                "📊 Текущее состояние: 🟢 <b>NORMAL</b>\n"
                "✅ <b>Сверка позиций успешно пройдена!</b>\n"
                "Торговый цикл запущен в штатном режиме."
            )
        else:
            text = (
                "🛡️ <b>Управление рисками (SRE Kill Switch)</b>\n\n"
                "📊 Текущее состояние: 🟡 <b>SAFE_MODE</b>\n"
                f"❌ <b>Ошибка сверки! Обнаружено расхождение:</b>\n"
                f"<code>{error_details}</code>\n\n"
                "Устраните расхождения вручную или нажмите кнопку ниже для синхронизации БД."
            )
    finally:
        if hasattr(exchange, "close"):
            await exchange.close()

    await query.message.edit_text(text, reply_markup=get_risk_keyboard())


async def risk_kill_handler(query: CallbackQuery, redis: Redis):
    await query.answer("Экстренный стоп активирован!")
    manager = KillSwitchManager(redis)
    await manager.set_state(
        KillSwitchState.KILLED, "MANUAL", "Экстренная ручная остановка администратором."
    )

    text = (
        "🛡️ <b>Управление рисками (SRE Kill Switch)</b>\n\n"
        "📊 Текущее состояние: 🔴 <b>KILLED</b>\n"
        "🚨 <b>Торговый цикл полностью заблокирован вручную.</b>\n"
        "Для разблокировки требуется устранить причины и запустить сверку."
    )
    await query.message.edit_text(text, reply_markup=get_risk_keyboard())


async def risk_normal_handler(query: CallbackQuery, redis: Redis):
    await query.answer("Режим NORMAL активирован")
    manager = KillSwitchManager(redis)
    await manager.set_state(
        KillSwitchState.NORMAL, "MANUAL", "Принудительный запуск администратором."
    )

    text = (
        "🛡️ <b>Управление рисками (SRE Kill Switch)</b>\n\n"
        "📊 Текущее состояние: 🟢 <b>NORMAL</b>\n"
        "⚠️ <b>Внимание: Режим NORMAL активирован принудительно (без сверки позиций).</b>"
    )
    await query.message.edit_text(text, reply_markup=get_risk_keyboard())


async def risk_sync_db_handler(
    query: CallbackQuery, session: AsyncSession, redis: Redis
):
    """Механизм Position Recovery: принудительно синхронизирует базу данных под состояние биржи"""
    await query.answer("Синхронизирую БД...")

    manager = KillSwitchManager(redis)
    settings = get_settings()
    symbols = [config[0] for config in settings.ACTIVE_CONFIGS]

    if settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET:
        exchange = BinanceExchange(
            settings.BINANCE_API_KEY,
            settings.BINANCE_API_SECRET,
            settings.BINANCE_TESTNET,
        )
    else:
        exchange = PaperExchange(session)

    try:
        repo = PaperTradingRepository(session)
        closed_count = 0

        # Сверяем каждый актив и закрываем фантомные сделки в БД
        for symbol in symbols:
            ex_pos = await exchange.get_position(symbol)
            db_pos = await repo.get_active_trade(symbol)

            # Если на бирже пусто, а в БД висит сделка — принудительно гасим её в БД
            if ex_pos is None and db_pos is not None:
                await repo.close_trade(db_pos, exit_price=db_pos.entry_price, pnl=0.0)
                closed_count += 1

        # Возвращаем статус NORMAL
        await manager.set_state(KillSwitchState.NORMAL)

        text = (
            "🛡️ <b>Управление рисками (SRE Kill Switch)</b>\n\n"
            "📊 Текущее состояние: 🟢 <b>NORMAL</b>\n"
            f"🔧 <b>Синхронизация завершена. Принудительно закрыто фантомных сделок в БД: {closed_count}.</b>\n"
            "База данных успешно приведена в соответствие с биржей. Бот запущен."
        )
    finally:
        if hasattr(exchange, "close"):
            await exchange.close()

    await query.message.edit_text(text, reply_markup=get_risk_keyboard())


def register_handlers():
    router.callback_query.register(
        admin_cancel_handler, F.data == "admin:cancel", IsAdmin()
    )
    router.callback_query.register(
        risk_reset_handler, F.data == "admin:risk:reset", IsAdmin()
    )
    router.callback_query.register(
        risk_kill_handler, F.data == "admin:risk:kill", IsAdmin()
    )
    router.callback_query.register(
        risk_normal_handler, F.data == "admin:risk:normal", IsAdmin()
    )
    # Регистрируем хэндлер восстановления позиций
    router.callback_query.register(
        risk_sync_db_handler, F.data == "admin:risk:sync_db", IsAdmin()
    )
