from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.filters.check_admin import IsAdmin
from src.risk import KillSwitchManager, KillSwitchState, reconcile_positions
from src.exchange.paper import PaperExchange
from src.core.config import get_settings

router = Router()


def get_risk_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(
            text="🔄 Сбросить и сверить", callback_data="admin:risk:reset"
        ),
    )
    builder.row(
        InlineKeyboardButton(
            text="🚨 Экстренный Стоп", callback_data="admin:risk:kill"
        ),
        InlineKeyboardButton(
            text="🟢 Активировать NORMAL", callback_data="admin:risk:normal"
        ),
    )
    return builder.as_markup()


async def admin_cancel_handler(query: CallbackQuery):
    await query.message.edit_text("❌ Операция отменена.")


async def risk_reset_handler(query: CallbackQuery, session: AsyncSession, redis: Redis):
    """Сбрасывает блокировку и запускает мгновенную сверку позиций"""
    await query.answer("Запускаю сверку...")

    manager = KillSwitchManager(redis)
    settings = get_settings()
    symbols = [config[0] for config in settings.ACTIVE_CONFIGS]

    # 1. Сбрасываем статус, чтобы разрешить временную сверку
    await manager.set_state(KillSwitchState.NORMAL)

    # 2. Инициализируем наш симулятор и сверяем позиции копейка в копейку
    exchange = PaperExchange(session)
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
            "Устраните расхождения на бирже или в БД и повторите сверку."
        )

    await query.message.edit_text(text, reply_markup=get_risk_keyboard())


async def risk_kill_handler(query: CallbackQuery, redis: Redis):
    """Экстренно замораживает всю торговлю"""
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
    """Принудительно разблокирует бота без сверки"""
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


def register_handlers():
    router.callback_query.register(
        admin_cancel_handler, F.data == "admin:cancel", IsAdmin()
    )
    # Регистрируем новые callback-обработчики
    router.callback_query.register(
        risk_reset_handler, F.data == "admin:risk:reset", IsAdmin()
    )
    router.callback_query.register(
        risk_kill_handler, F.data == "admin:risk:kill", IsAdmin()
    )
    router.callback_query.register(
        risk_normal_handler, F.data == "admin:risk:normal", IsAdmin()
    )
