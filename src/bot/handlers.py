from html import escape

from aiogram import Router
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from src.config import get_settings
from src.db import get_open_trades, get_recent_closed_trades
from src.bot.keyboards import main_keyboard
from src.exchange.base import BaseExchange

router = Router()


@router.message(CommandStart())
async def start_cmd(message: Message):
    settings = get_settings()
    await message.answer(
        f"🤖 <b>MarketMind Quant Bot</b>\n\n"
        f"Режим торговли: <code>{settings.TRADING_MODE.upper()}</code>\n"
        f"Используйте кнопки меню для мониторинга.",
        reply_markup=main_keyboard(),
        parse_mode="HTML"
    )


@router.message(Command("status"))
async def status_cmd(message: Message, exchange: BaseExchange):
    settings = get_settings()
    try:
        balance = await exchange.get_balance()
        bal_text = f"💵 Свободно: <code>${balance['free']:.2f}</code> | Всего: <code>${balance['total']:.2f}</code>"
    except Exception as error:
        bal_text = f"⚠️ Ошибка баланса: {escape(str(error))}"

    configs = ", ".join(
        f"{escape(symbol)} ({escape(timeframe)})"
        for symbol, timeframe in settings.ACTIVE_CONFIGS
    )
    await message.answer(
        f"📊 <b>Статус Системы</b>\n\n"
        f"Режим: <code>{settings.TRADING_MODE.upper()}</code>\n"
        f"{bal_text}\n"
        f"Активные пары: <code>{configs}</code>",
        parse_mode="HTML"
    )


@router.message(Command("positions"))
async def positions_cmd(message: Message):
    trades = await get_open_trades()

    if not trades:
        await message.answer(
            "📭 Активных открытых позиций нет.",
            parse_mode="HTML",
        )
        return

    lines = ["🚀 <b>Открытые Позиции:</b>\n"]
    for trade in trades:
        lines.append(
            f"• <b>{escape(trade.symbol)}</b> ({escape(trade.side)})\n"
            f"  Объем: <code>{trade.amount}</code> @ <code>${trade.entry_price:.2f}</code>\n"
            f"  SL: <code>${trade.sl_price or 0:.2f}</code> | TP: <code>${trade.tp_price or 0:.2f}</code>"
        )
    await message.answer("\n\n".join(lines), parse_mode="HTML")


@router.message(Command("trades"))
async def trades_cmd(message: Message):
    trades = await get_recent_closed_trades()

    if not trades:
        await message.answer(
            "📭 История закрытых сделок пуста.",
            parse_mode="HTML",
        )
        return

    total_pnl = sum(trade.pnl or 0.0 for trade in trades)
    lines = [f"📈 <b>Последние 10 сделок (PnL: ${total_pnl:+.2f}):</b>\n"]
    for trade in trades:
        pnl_str = f"${trade.pnl:+.2f}" if trade.pnl is not None else "N/A"
        lines.append(
            f"• <b>{escape(trade.symbol)}</b> ({escape(trade.side)}) | PnL: <code>{pnl_str}</code>"
        )
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("risk"))
async def risk_cmd(message: Message):
    settings = get_settings()
    await message.answer(
        f"🛡️ <b>Настройки Риск-Менеджмента</b>\n\n"
        f"Макс. % банка на сделку: <code>{settings.RISK_MAX_ALLOCATION_PCT * 100}%</code>\n"
        f"Макс. дневная просадка: <code>{settings.RISK_MAX_DAILY_LOSS_PCT * 100}%</code>\n"
        f"Circuit Breaker (серия убытков): <code>{settings.RISK_CONSECUTIVE_LOSSES_LIMIT}</code>\n"
        f"Дефолтный SL / TP: <code>{settings.DEFAULT_SL_PCT * 100}% / {settings.DEFAULT_TP_PCT * 100}%</code>",
        parse_mode="HTML"
    )
