"""Read-only operational Telegram commands backed by exchange projections."""

import json
import os
import pickle
from datetime import datetime, timezone

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.redis import get_redis
from src.db.models import (BalanceSnapshot, ExchangeEvent, ExchangeFill,
                           PositionSnapshot, ReconciliationRun, Trade)
from src.filters.check_admin import IsAdmin
from src.risk.kill_switch import KillSwitchManager, reconcile_positions

router = Router()


def _money(value) -> str:
    return "—" if value is None else f"${float(value):,.2f}"


def _live_exchange(settings):
    from src.exchange.binance import BinanceExchange
    return BinanceExchange(
        api_key=settings.BINANCE_API_KEY,
        secret=settings.BINANCE_API_SECRET,
        testnet=settings.BINANCE_TESTNET,
    )


@router.message(Command("portfolio"))
async def portfolio(message: Message, session: AsyncSession):
    settings = get_settings()
    if settings.LIVE_TRADING:
        exchange = _live_exchange(settings)
        try:
            balance = await exchange.get_balance()
            await message.answer(
                "<b>Portfolio · Binance Futures</b>\n"
                f"Available: <code>{_money(balance.get('free'))}</code>\n"
                f"Total: <code>{_money(balance.get('total'))}</code>"
            )
        finally:
            await exchange.close()
        return
    rows = (await session.execute(select(BalanceSnapshot).where(
        BalanceSnapshot.environment == settings.TRADING_MODE))).scalars().all()
    if not rows:
        await message.answer("Binance snapshot ещё не получен. Запустите /reconcile.")
        return
    text = ["<b>Portfolio · Binance Futures</b>"]
    for row in rows:
        text.append(f"{row.asset}: wallet {_money(row.wallet_balance)} · available {_money(row.available_balance)}")
    await message.answer("\n".join(text))


@router.message(Command("positions"))
async def positions(message: Message, session: AsyncSession):
    settings = get_settings()
    if settings.LIVE_TRADING:
        exchange = _live_exchange(settings)
        try:
            lines = ["<b>Open positions · Binance</b>"]
            for symbol, _ in settings.ACTIVE_CONFIGS:
                position = await exchange.get_position(symbol)
                if not position or not position.get("amount"):
                    continue
                orders = await exchange.get_open_orders(symbol)
                has_sl = any("STOP" in str(order.get("type", "")).upper() for order in orders)
                has_tp = any("TAKE_PROFIT" in str(order.get("type", "")).upper() for order in orders)
                lines.append(
                    f"{symbol} · {position.get('side')} {float(position['amount']):.6f}\n"
                    f"Entry {_money(position.get('entry_price'))} · Mark {_money(position.get('mark_price'))} · "
                    f"uPnL {_money(position.get('unrealized_pnl'))}\n"
                    f"SL {'✅' if has_sl else '—'} · TP {'✅' if has_tp else '—'}"
                )
            await message.answer("\n\n".join(lines) if len(lines) > 1 else "<b>Open positions · Binance</b>\nНет открытых позиций.")
        finally:
            await exchange.close()
        return
    rows = (await session.execute(select(PositionSnapshot).where(
        PositionSnapshot.environment == settings.TRADING_MODE,
        PositionSnapshot.amount != 0))).scalars().all()
    if not rows:
        await message.answer("<b>Open positions</b>\nНет открытых позиций.")
        return
    text = ["<b>Open positions · Binance</b>"]
    for p in rows:
        trade = (await session.execute(select(Trade).where(Trade.symbol == p.symbol, Trade.status == "OPEN", Trade.environment == settings.TRADING_MODE))).scalar_one_or_none()
        protection = "SL — · TP —" if trade is None else f"SL {'✅' if trade.sl_price else '—'} · TP {'✅' if trade.tp_price else '—'}"
        text.append(f"{p.symbol} · {p.side} {float(p.amount):.6f}\nEntry {_money(p.entry_price)} · Mark {_money(p.mark_price)} · uPnL {_money(p.unrealized_pnl)}\n{protection}")
    await message.answer("\n\n".join(text))


@router.message(Command("trades"))
async def trades(message: Message, session: AsyncSession):
    settings = get_settings()
    fills = (await session.execute(select(ExchangeFill).where(ExchangeFill.environment == settings.TRADING_MODE).order_by(desc(ExchangeFill.created_at)).limit(10))).scalars().all()
    closed = (await session.execute(select(Trade).where(
        Trade.environment == settings.TRADING_MODE, Trade.status == "CLOSED"
    ).order_by(desc(Trade.exit_time)).limit(5))).scalars().all()
    if not fills and not closed:
        await message.answer("<b>Последние fills и сделки</b>\nНет данных.")
        return
    lines = ["<b>Последние fills · Binance</b>"]
    for f in fills:
        lines.append(f"{f.symbol} {f.side} {float(f.amount):.6f} @ {_money(f.price)} · fee {_money(f.commission)} · rPnL {_money(f.realized_pnl)} · order <code>{f.exchange_order_id or '—'}</code>")
    if closed:
        lines.append("\n<b>Закрытые сделки</b>")
        for trade in closed:
            lines.append(f"{trade.symbol} · PnL {_money(trade.pnl)} · entry {_money(trade.entry_price)} → exit {_money(trade.exit_price)} · order <code>{trade.exit_order_id or '—'}</code>")
    await message.answer("\n".join(lines))


def _model_details(symbol: str) -> tuple[str, dict]:
    settings = get_settings()
    match = next(((s, tf) for s, tf in settings.ACTIVE_CONFIGS if s.replace("/", "") == symbol.upper()), None)
    if not match:
        return "", {}
    path = settings.get_model_path(*match)
    if not os.path.exists(path):
        return path, {}
    with open(path, "rb") as artifact_file:
        artifact = pickle.load(artifact_file)
    return path, artifact if isinstance(artifact, dict) else {}


@router.message(Command("models"))
async def models(message: Message):
    settings = get_settings()
    lines = ["<b>Champion models</b>"]
    for symbol, timeframe in settings.ACTIVE_CONFIGS:
        path, artifact = _model_details(symbol.replace("/", ""))
        if not artifact:
            lines.append(f"{symbol} {timeframe}: отсутствует")
            continue
        lines.append(f"{symbol} {timeframe}: <code>{artifact.get('model_id', os.path.basename(path))}</code>")
    await message.answer("\n".join(lines))


@router.message(Command("model"))
async def model(message: Message, command: CommandObject):
    symbol = (command.args or "").strip().upper()
    if not symbol:
        await message.answer("Использование: <code>/model ETHUSDT</code>")
        return
    path, artifact = _model_details(symbol)
    if not artifact:
        await message.answer(f"Champion-модель для {symbol} не найдена.")
        return
    # Keep evaluation domains visibly separate; never mix offline and trading PnL.
    offline = artifact.get("offline_metrics") or artifact.get("metrics") or {}
    shadow = artifact.get("shadow_live_metrics") or {}
    trading = artifact.get("trading_metrics") or {}
    def dumped(value): return json.dumps(value, ensure_ascii=False, default=str) or "нет данных"
    await message.answer(
        f"<b>{symbol} · {artifact.get('model_id', os.path.basename(path))}</b>\n\n"
        f"<b>offline</b> (walk-forward / holdout)\n<code>{dumped(offline)}</code>\n\n"
        f"<b>shadow/live predictions</b> (завершённые прогнозы)\n<code>{dumped(shadow)}</code>\n\n"
        f"<b>trading</b> (PnL, Sharpe, drawdown, комиссии, slippage)\n<code>{dumped(trading)}</code>"
    )


@router.message(Command("health"))
async def health(message: Message, session: AsyncSession):
    redis = await get_redis()
    state, reason, _ = await KillSwitchManager(redis).get_state()
    latest_event = (await session.execute(select(ExchangeEvent).order_by(desc(ExchangeEvent.received_at)).limit(1))).scalar_one_or_none()
    latest_reconcile = (await session.execute(select(ReconciliationRun).order_by(desc(ReconciliationRun.created_at)).limit(1))).scalar_one_or_none()
    now = datetime.now(timezone.utc)
    ws = "нет событий" if latest_event is None else f"{(now - latest_event.received_at).total_seconds():.0f}s назад"
    rec = "нет запусков" if latest_reconcile is None else f"{latest_reconcile.status} · {(now - latest_reconcile.created_at).total_seconds():.0f}s назад"
    await message.answer(f"<b>Health</b>\nWS freshness: <code>{ws}</code>\nReconciliation: <code>{rec}</code>\nKill switch: <code>{state}</code>{f' · {reason}' if reason else ''}\nData freshness: <code>{ws}</code>")


@router.message(Command("reconcile"), IsAdmin())
async def reconcile(message: Message, session: AsyncSession):
    settings = get_settings()
    if not settings.LIVE_TRADING:
        await message.answer("Ручная сверка доступна только в testnet/mainnet.")
        return
    from src.exchange.binance import BinanceExchange
    exchange = BinanceExchange(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET, testnet=settings.BINANCE_TESTNET)
    try:
        manager = KillSwitchManager(await get_redis())
        ok, details = await reconcile_positions(exchange, session, [s for s, _ in settings.ACTIVE_CONFIGS], manager, environment=settings.TRADING_MODE, verify_protection=True)
        await message.answer(f"<b>Reconciliation {'OK' if ok else 'UNSAFE'}</b>\n<code>{details or 'Сверка завершена без расхождений.'}</code>")
    finally:
        await exchange.close()


def register_handlers():
    return None
