import asyncio
import sys
import os
import pandas as pd
import shutil
import pickle
from datetime import datetime, timezone

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.redis import RedisStorage
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.db import engine, Base, AsyncSessionFactory
from src.core.redis import get_redis, close_redis
from src.core.router_manager import setup_routers
from src.filters.chat_type import ChatTypeFilter
from src.middlewares.db import DBSessionMiddleware
from src.middlewares.logger import LoggerMiddleware
from src.middlewares.rate_limit import RateLimitMiddleware
from src.middlewares.redis import RedisMiddleware

# Мягкая SRE интеграция
try:
    from nexus_sdk import NexusSDK

    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    logger.warning("NexusSDK не установлен. Запуск без Nexus SRE.")


async def check_and_rollback_model(
    session: AsyncSession, bot: Bot, symbol: str, timeframe: str
):
    """
    Фоновая проверка качества живых сделок по конкретному активу (Quest 8 & 9).
    Если метрики последних N сделок деградировали ниже порогов,
    автоматически откатывает рабочую модель на последний успешный бэкап.
    """
    import glob
    from src.crud.paper import PaperTradingRepository
    from src.strategy.signals import calculate_strategy_metrics

    settings = get_settings()
    model_path = settings.get_model_path(symbol, timeframe)

    # 1. Проверяем кулдаун отката по этой паре в Redis
    try:
        redis = await get_redis()
        cooldown = await redis.get(f"model_rollback_cooldown:{symbol}:{timeframe}")
        if cooldown:
            return
    except Exception as re_err:
        logger.error(f"Ошибка проверки кулдауна в Redis для {symbol}: {re_err}")

    repo = PaperTradingRepository(session)
    closed_trades = await repo.get_closed_trades(
        symbol, limit=settings.ROLLBACK_CHECK_WINDOW
    )

    if len(closed_trades) < settings.ROLLBACK_CHECK_WINDOW:
        return

    trade_returns = []
    for t in closed_trades:
        is_short = t.is_short

        if is_short:
            ret = (t.entry_price - t.exit_price) / t.entry_price
        else:
            ret = (t.exit_price - t.entry_price) / t.entry_price
        trade_returns.append(ret)

    metrics = calculate_strategy_metrics(trade_returns)

    win_rate = metrics["win_rate"]
    max_dd = metrics["max_drawdown"]

    is_degraded = (
        win_rate < settings.ROLLBACK_WIN_RATE_THRESHOLD
        or max_dd > settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD
    )

    if is_degraded:
        os_dir = os.path.dirname(model_path)
        clean_symbol = symbol.replace("/", "").replace(":", "")
        clean_tf = timeframe.replace("/", "")
        backup_pattern = os.path.join(
            os_dir, f"lgbm_{clean_symbol}_{clean_tf}_backup_*.pkl"
        )
        backup_files = glob.glob(backup_pattern)

        if not backup_files:
            msg = (
                f"🚨 [CRITICAL SRE] Живые показатели по {symbol} деградировали!\n"
                f"Win Rate: <code>{win_rate:.1%}</code> (порог: {settings.ROLLBACK_WIN_RATE_THRESHOLD:.1%})\n"
                f"Max Drawdown: <code>{max_dd:.1%}</code> (порог: {settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD:.1%})\n\n"
                f"⚠️ <b>Откат невозможен:</b> бэкапы не найдены в папке {os_dir}!"
            )
            logger.critical(msg)
            for admin_id in settings.ADMIN_IDS:
                try:
                    await bot.send_message(chat_id=admin_id, text=msg)
                except Exception as e:
                    logger.error(f"Не удалось отправить алерт админу {admin_id}: {e}")
            return

        backup_files.sort(key=os.path.getmtime, reverse=True)
        best_backup = backup_files[0]

        try:
            shutil.copy(best_backup, model_path)

            try:
                await redis.setex(
                    f"model_rollback_cooldown:{symbol}:{timeframe}", 86400, "1"
                )
            except Exception as re_err:
                logger.error(
                    f"Не удалось записать кулдаун отката для {symbol}: {re_err}"
                )

            msg = (
                f"🚨 [CRITICAL SRE] Живые показатели по {symbol} ДЕГРАДИРОВАЛИ!\n"
                f"Win Rate: <code>{win_rate:.1%}</code> (порог: {settings.ROLLBACK_WIN_RATE_THRESHOLD:.1%})\n"
                f"Max Drawdown: <code>{max_dd:.1%}</code> (порог: {settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD:.1%})\n\n"
                f"🔄 <b>Автоматический откат выполнен по {symbol}!</b>\n"
                f"Рабочая модель заменена на стабильный бэкап:\n"
                f"<code>{os.path.basename(best_backup)}</code>\n\n"
                f"⏱ Блокировка проверок на 24 часа."
            )
            logger.warning(msg)

            for admin_id in settings.ADMIN_IDS:
                try:
                    await bot.send_message(chat_id=admin_id, text=msg)
                except Exception as e:
                    logger.error(
                        f"Не удалось уведомить админа {admin_id} об откате {symbol}: {e}"
                    )

        except Exception as err:
            logger.error(f"Ошибка копирования при откате {symbol}: {err}")


async def paper_trading_loop(bot: Bot, symbol: str, timeframe: str):
    """
    Асинхронная фоновая служба бумажной торговли для конкретного актива (Quest 9).
    """
    from src.data.collector import DataCollector
    from src.models.predictor import Predictor
    from src.paper_trading.engine import PaperTradingEngine
    from src.crud.kline import KlineRepository

    settings = get_settings()
    model_path = settings.get_model_path(symbol, timeframe)
    logger.info(f"Фоновая служба Paper Trading для {symbol} ({timeframe}) запущена.")

    while True:
        try:
            # Опрашиваем биржу раз в 1 час
            await asyncio.sleep(3600)

            async with AsyncSessionFactory() as session:
                async with DataCollector(session) as collector:
                    await collector.fetch_and_save_klines(symbol, timeframe, limit=5)

                repo = KlineRepository(session)
                klines = await repo.get_klines(symbol, timeframe, limit=50)

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

                if not os.path.exists(model_path):
                    continue

                predictor = Predictor(model_path)
                engine_pt = PaperTradingEngine(session)

                log_msg = await engine_pt.process_market_update(
                    symbol=symbol,
                    timeframe=timeframe,
                    latest_candles=df,
                    predictor=predictor,
                )

                if log_msg:
                    # 1. Отправляем ВСЕ логи сделок администраторам
                    for admin_id in settings.ADMIN_IDS:
                        try:
                            await bot.send_message(chat_id=admin_id, text=log_msg)
                        except Exception as e:
                            logger.error(
                                f"Не удалось отправить уведомление админу {admin_id}: {e}"
                            )

                    # 2. Обычным подписчикам отправляем ТОЛЬКО закрытие сделок
                    is_closure = any(
                        log_msg.startswith(prefix) for prefix in ["🔴", "🟢", "🔵"]
                    )
                    if is_closure:
                        from src.crud.user import UserRepository

                        user_repo = UserRepository(session)
                        subscribed_users = await user_repo.get_all_subscribed()

                        for u in subscribed_users:
                            if u.user_id in settings.ADMIN_IDS:
                                continue
                            try:
                                await bot.send_message(chat_id=u.user_id, text=log_msg)
                            except Exception as e:
                                logger.error(
                                    f"Не удалось отправить сигнал подписчику {u.user_id}: {e}"
                                )

                        # 3. Запускаем SRE проверку на деградацию модели
                        try:
                            await check_and_rollback_model(
                                session, bot, symbol, timeframe
                            )
                        except Exception as rollback_err:
                            logger.error(
                                f"Ошибка при проверке/откате деградации модели {symbol}: {rollback_err}"
                            )

        except Exception as e:
            logger.error(f"Ошибка в цикле бумажной торговли для {symbol}: {e}")
            await asyncio.sleep(60)


async def retrain_loop(bot: Bot, symbol: str, timeframe: str):
    """
    Фоновая служба периодического переобучения модели для конкретного актива (Quest 9).
    """
    from src.core.db import AsyncSessionFactory
    from src.crud.kline import KlineRepository
    from src.datasets.build import build_and_save_dataset
    from src.models.baseline import run_baseline_experiment
    from src.models.train import run_lgbm_experiment

    settings = get_settings()
    model_path = settings.get_model_path(symbol, timeframe)
    logger.info(f"Фоновая служба автообучения для {symbol} ({timeframe}) запущена.")

    while True:
        try:
            await asyncio.sleep(settings.RETRAIN_INTERVAL_SECONDS)

            async with AsyncSessionFactory() as session:
                kline_repo = KlineRepository(session)
                klines = await kline_repo.get_klines(
                    symbol, timeframe, limit=settings.MIN_KLINES_FOR_TRAIN
                )
                if len(klines) < settings.MIN_KLINES_FOR_TRAIN:
                    logger.info(
                        f"[Retrain - {symbol}] Недостаточно данных: {len(klines)}/{settings.MIN_KLINES_FOR_TRAIN}"
                    )
                    continue

                version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")

                parquet_path = await build_and_save_dataset(
                    session,
                    symbol=symbol,
                    timeframe=timeframe,
                    version=version,
                    horizon=settings.LABEL_HORIZON,
                    threshold=settings.LABEL_THRESHOLD,
                )
                json_path = parquet_path.replace(".parquet", ".json")

                baseline_result = await run_baseline_experiment(
                    session,
                    parquet_path,
                    json_path,
                    train_size=settings.TRAIN_SIZE,
                    test_size=settings.TEST_SIZE,
                )
                lgbm_result = await run_lgbm_experiment(
                    session,
                    parquet_path,
                    json_path,
                    train_size=settings.TRAIN_SIZE,
                    test_size=settings.TEST_SIZE,
                    models_dir="models/staging",
                    baseline_f1=baseline_f1,
                )

                baseline_f1 = baseline_result["metrics"]["f1"]
                new_f1 = lgbm_result["metrics"]["f1"]

                if new_f1 <= baseline_f1:
                    msg = (
                        f"⚠️ [Retrain v{version} - {symbol}] Новая модель НЕ превзошла baseline "
                        f"(LGBM f1={new_f1:.3f} vs baseline f1={baseline_f1:.3f}). "
                        f"В продакшн НЕ продвигается."
                    )
                    logger.warning(msg)
                else:
                    os_dir = os.path.dirname(model_path)
                    if os_dir:
                        os.makedirs(os_dir, exist_ok=True)

                    # --- АВТОМАТИЧЕСКАЯ КАЛИБРОВКА И ВШИВАНИЕ РИСКОВ В АРТЕФАКТ ---
                    try:
                        from scripts.calibrate import get_best_calibration

                        # Калибруем по временной модели, пока она лежит в staging
                        best_sl, best_tp, cal_report = await get_best_calibration(
                            symbol,
                            timeframe,
                            custom_model_path=lgbm_result["model_path"],
                        )

                        # Открываем артефакт в staging, обновляем параметры калибровки
                        with open(lgbm_result["model_path"], "rb") as f:
                            artifact = pickle.load(f)

                        artifact["calibration"] = {
                            "sl_pct": best_sl,
                            "tp_pct": best_tp,
                            "calibrated_at": datetime.now(timezone.utc).isoformat(),
                        }

                        # Перезаписываем артефакт в staging
                        with open(lgbm_result["model_path"], "wb") as f:
                            pickle.dump(artifact, f)

                        msg = (
                            f"✅ [Retrain v{version} - {symbol}] Модель обновлена в продакшне.\n"
                            f"accuracy={lgbm_result['metrics']['accuracy']:.3f}, "
                            f"f1={new_f1:.3f} (baseline f1={baseline_f1:.3f})\n\n"
                            f"{cal_report}"
                        )
                        logger.info(
                            f"[Retrain v{version} - {symbol}] Автокалибровка завершена. SL={best_sl:.1%}, TP={best_tp:.1%}"
                        )
                    except Exception as cal_err:
                        logger.error(f"Ошибка автокалибровки для {symbol}: {cal_err}")
                        msg = (
                            f"✅ [Retrain v{version} - {symbol}] Модель обновлена в продакшне.\n"
                            f"accuracy={lgbm_result['metrics']['accuracy']:.3f}, "
                            f"f1={new_f1:.3f} (baseline f1={baseline_f1:.3f})\n\n"
                            f"⚠️ Автокалибровка завершилась с ошибкой: {cal_err}"
                        )

                    # --- Создаем бэкап старой модели перед заменой (Quest 8 & 9) ---
                    if os.path.exists(model_path):
                        clean_symbol = symbol.replace("/", "").replace(":", "")
                        clean_tf = timeframe.replace("/", "")
                        backup_filename = f"lgbm_{clean_symbol}_{clean_tf}_backup_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}.pkl"
                        backup_path = os.path.join(os_dir, backup_filename)
                        try:
                            shutil.copy(model_path, backup_path)
                            logger.info(
                                f"[Retrain - {symbol}] Успешно создан бэкап старой стабильной модели: {backup_path}"
                            )
                        except Exception as copy_err:
                            logger.error(
                                f"Не удалось скопировать бэкап для {symbol}: {copy_err}"
                            )

                    # Копируем полностью укомплектованный артефакт в продакшн-папку
                    shutil.copy(lgbm_result["model_path"], model_path)

                    msg = (
                        f"✅ [Retrain v{version} - {symbol}] Модель обновлена в продакшне.\n"
                        f"accuracy={lgbm_result['metrics']['accuracy']:.3f}, "
                        f"f1={new_f1:.3f} (baseline f1={baseline_f1:.3f})\n"
                    )
                    logger.info(msg)

                    # --- АВТОМАТИЧЕСКАЯ КАЛИБРОВКА (Способ 2) ---
                    try:
                        from scripts.calibrate import get_best_calibration

                        best_sl, best_tp, cal_report = await get_best_calibration(
                            symbol, timeframe
                        )
                        msg += f"\n{cal_report}"
                        logger.info(
                            f"[Retrain v{version} - {symbol}] Автокалибровка завершена: SL={best_sl:.1%}, TP={best_tp:.1%}"
                        )
                    except Exception as cal_err:
                        logger.error(f"Ошибка автокалибровки для {symbol}: {cal_err}")
                        msg += f"\n\n⚠️ Автокалибровка завершилась с ошибкой для {symbol}: {cal_err}"

                for admin_id in settings.ADMIN_IDS:
                    try:
                        await bot.send_message(chat_id=admin_id, text=msg)
                    except Exception as e:
                        logger.error(f"Не удалось уведомить админа {admin_id}: {e}")

        except Exception as e:
            logger.error(f"Ошибка в цикле автообучения для {symbol}: {e}")
            await asyncio.sleep(60)


async def on_startup(bot: Bot):
    bot_info = await bot.get_me()
    logger.info(
        f"Bot started: {bot_info.full_name} (@{bot_info.username}, id={bot_info.id})"
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    settings = get_settings()

    # Запускаем фоновые службы параллельно для каждого настроенного актива (Quest 9)
    for symbol, timeframe in settings.ACTIVE_CONFIGS:
        logger.info(
            f"[*] Запуск фоновых процессов параллельно для {symbol} ({timeframe})"
        )
        asyncio.create_task(paper_trading_loop(bot, symbol, timeframe))
        asyncio.create_task(retrain_loop(bot, symbol, timeframe))


async def on_shutdown(bot: Bot):
    global nexus
    logger.info("Shutting down...")

    if NEXUS_AVAILABLE and nexus:
        try:
            await nexus.close()
            logger.info("Nexus SRE сессия успешно завершена.")
        except Exception as e:
            logger.error(f"Ошибка при остановке Nexus SDK: {e}")

    await close_redis()
    await engine.dispose()
    await bot.session.close()


async def main():
    global nexus
    settings = get_settings()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level=settings.LOG_LEVEL,
        colorize=True,
    )
    logger.add(
        "logs/bot.log",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        level="DEBUG",
    )

    redis = await get_redis()
    storage = RedisStorage(redis=redis)

    bot = Bot(
        token=settings.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher(storage=storage)

    # Инициализируем SRE
    if NEXUS_AVAILABLE:
        app_secret = os.getenv("NEXUS_APP_SECRET")
        if app_secret:
            nexus = NexusSDK(
                endpoint_url=os.getenv(
                    "NEXUS_ENDPOINT_URL", "http://nexus-webhook:8000/events/app"
                ),
                app_secret=app_secret,
                project_name=os.getenv("NEXUS_PROJECT_NAME", "binance_quant_bot"),
            )
            nexus.register_aiogram_error_handler(dp)
            nexus.start_heartbeat(interval_seconds=15)
            logger.info("Nexus SRE мониторинг успешно инициализирован.")
        else:
            logger.warning(
                "NEXUS_APP_SECRET не задан. Запуск без интеграции с Nexus SRE."
            )

    dp.update.outer_middleware(LoggerMiddleware())
    dp.update.outer_middleware(RedisMiddleware(redis))
    dp.update.outer_middleware(DBSessionMiddleware())
    dp.update.outer_middleware(RateLimitMiddleware(redis))

    dp.message.filter(ChatTypeFilter(chat_type="private"))

    router = setup_routers()
    dp.include_router(router)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await on_shutdown(bot)


def cli():
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


if __name__ == "__main__":
    cli()
