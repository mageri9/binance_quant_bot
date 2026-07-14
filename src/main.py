import asyncio
import sys
import os
import pandas as pd
import shutil
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

# Мягкая интеграция с Nexus SRE SDK
try:
    from nexus_sdk import NexusSDK

    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    logger.warning("NexusSDK не установлен в окружении. Запуск без Nexus SRE.")


async def check_and_rollback_model(session: AsyncSession, bot: Bot):
    """
    Фоновая проверка качества живых сделок (Quest 8).
    Если метрики последних N сделок деградировали ниже порогов,
    автоматически откатывает рабочую модель на последний успешный бэкап.
    """
    import glob
    from src.crud.paper import PaperTradingRepository
    from src.strategy.signals import calculate_strategy_metrics

    settings = get_settings()

    # 1. Проверяем кулдаун отката в Redis (чтобы не откатывать модель на каждой сделке подряд)
    try:
        redis = await get_redis()
        cooldown = await redis.get("model_rollback_cooldown")
        if cooldown:
            return
    except Exception as re_err:
        logger.error(f"Ошибка проверки кулдауна отката в Redis: {re_err}")

    repo = PaperTradingRepository(session)
    # Загружаем последние сделки
    closed_trades = await repo.get_closed_trades(
        "BTC/USDT", limit=settings.ROLLBACK_CHECK_WINDOW
    )

    if len(closed_trades) < settings.ROLLBACK_CHECK_WINDOW:
        # Недостаточно статистики для анализа
        return

    # Рассчитываем доходность
    trade_returns = [
        (t.exit_price - t.entry_price) / t.entry_price for t in closed_trades
    ]
    metrics = calculate_strategy_metrics(trade_returns)

    win_rate = metrics["win_rate"]
    max_dd = metrics["max_drawdown"]

    # Условие деградации: Win Rate ниже нормы ИЛИ просадка выше порога
    is_degraded = (
        win_rate < settings.ROLLBACK_WIN_RATE_THRESHOLD
        or max_dd > settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD
    )

    if is_degraded:
        os_dir = os.path.dirname(settings.MODEL_PATH)
        backup_pattern = os.path.join(os_dir, "lgbm_BTCUSDT_1h_backup_*.pkl")
        backup_files = glob.glob(backup_pattern)

        if not backup_files:
            msg = (
                f"🚨 [CRITICAL SRE] Живые показатели модели деградировали!\n"
                f"Win Rate: <code>{win_rate:.1%}</code> (порог: {settings.ROLLBACK_WIN_RATE_THRESHOLD:.1%})\n"
                f"Max Drawdown: <code>{max_dd:.1%}</code> (порог: {settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD:.1%})\n\n"
                f"⚠️ <b>Откат невозможен:</b> файлы резервных бэкапов не найдены в папке {os_dir}!"
            )
            logger.critical(msg)
            for admin_id in settings.ADMIN_IDS:
                try:
                    await bot.send_message(chat_id=admin_id, text=msg)
                except Exception as e:
                    logger.error(f"Не удалось отправить алерт админу {admin_id}: {e}")
            return

        # Сортируем бэкапы по времени изменения (самый свежий — первый)
        backup_files.sort(key=os.path.getmtime, reverse=True)
        best_backup = backup_files[0]

        # Выполняем автооткат файла модели
        try:
            shutil.copy(best_backup, settings.MODEL_PATH)

            # Устанавливаем кулдаун в Redis на 24 часа
            try:
                await redis.setex("model_rollback_cooldown", 86400, "1")
            except Exception as re_err:
                logger.error(f"Не удалось записать кулдаун отката в Redis: {re_err}")

            msg = (
                f"🚨 [CRITICAL SRE] Живые показатели модели ДЕГРАДИРОВАЛИ!\n"
                f"Win Rate: <code>{win_rate:.1%}</code> (порог: {settings.ROLLBACK_WIN_RATE_THRESHOLD:.1%})\n"
                f"Max Drawdown: <code>{max_dd:.1%}</code> (порог: {settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD:.1%})\n\n"
                f"🔄 <b>Автоматический откат выполнен!</b>\n"
                f"Рабочая модель успешно заменена на последний стабильный бэкап:\n"
                f"<code>{os.path.basename(best_backup)}</code>\n\n"
                f"⏱ Включена блокировка проверок на 24 часа."
            )
            logger.warning(msg)

            for admin_id in settings.ADMIN_IDS:
                try:
                    await bot.send_message(chat_id=admin_id, text=msg)
                except Exception as e:
                    logger.error(
                        f"Не удалось отправить уведомление об откате админу {admin_id}: {e}"
                    )

        except Exception as err:
            logger.error(f"Ошибка при копировании резервной копии модели: {err}")


async def paper_trading_loop(bot: Bot):
    """
    Асинхронная фоновая служба бумажной торговли.
    """
    from src.data.collector import DataCollector
    from src.models.predictor import Predictor
    from src.paper_trading.engine import PaperTradingEngine
    from src.crud.kline import KlineRepository

    settings = get_settings()
    logger.info("Фоновая служба Paper Trading запущена.")

    while True:
        try:
            # Опрашиваем биржу раз в 1 час
            await asyncio.sleep(3600)

            async with AsyncSessionFactory() as session:
                collector = DataCollector(session)
                await collector.fetch_and_save_klines("BTC/USDT", "1h", limit=5)
                await collector.close()

                repo = KlineRepository(session)
                klines = await repo.get_klines("BTC/USDT", "1h", limit=50)

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

                if not os.path.exists(settings.MODEL_PATH):
                    continue

                predictor = Predictor(settings.MODEL_PATH)
                engine_pt = PaperTradingEngine(session)

                log_msg = await engine_pt.process_market_update(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    latest_candles=df,
                    predictor=predictor,
                )

                if log_msg:
                    # 1. Отправляем ВСЕ логи сделок администраторам (открытие, закрытие, отмена)
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

                        # 3. Запускаем SRE проверку на деградацию модели (Quest 8)
                        try:
                            await check_and_rollback_model(session, bot)
                        except Exception as rollback_err:
                            logger.error(
                                f"Ошибка при проверке/откате деградации модели: {rollback_err}"
                            )

        except Exception as e:
            logger.error(f"Ошибка в цикле бумажной торговли: {e}")
            await asyncio.sleep(60)


nexus = None


async def retrain_loop(bot: Bot):
    """
    Фоновая служба периодического переобучения модели.
    """
    from src.core.db import AsyncSessionFactory
    from src.crud.kline import KlineRepository
    from src.datasets.build import build_and_save_dataset
    from src.models.baseline import run_baseline_experiment
    from src.models.train import run_lgbm_experiment

    settings = get_settings()
    logger.info("Фоновая служба автообучения запущена.")

    while True:
        try:
            await asyncio.sleep(settings.RETRAIN_INTERVAL_SECONDS)

            async with AsyncSessionFactory() as session:
                kline_repo = KlineRepository(session)
                klines = await kline_repo.get_klines(
                    "BTC/USDT", "1h", limit=settings.MIN_KLINES_FOR_TRAIN
                )
                if len(klines) < settings.MIN_KLINES_FOR_TRAIN:
                    logger.info(
                        f"[Retrain] Недостаточно данных: {len(klines)}/{settings.MIN_KLINES_FOR_TRAIN}"
                    )
                    continue

                version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")

                parquet_path = await build_and_save_dataset(
                    session,
                    symbol="BTC/USDT",
                    timeframe="1h",
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
                )

                baseline_f1 = baseline_result["metrics"]["f1"]
                new_f1 = lgbm_result["metrics"]["f1"]

                if new_f1 <= baseline_f1:
                    msg = (
                        f"⚠️ [Retrain v{version}] Новая модель НЕ превзошла baseline "
                        f"(LGBM f1={new_f1:.3f} vs baseline f1={baseline_f1:.3f}). "
                        f"В продакшн НЕ продвигается."
                    )
                    logger.warning(msg)
                else:
                    os_dir = os.path.dirname(settings.MODEL_PATH)
                    if os_dir:
                        os.makedirs(os_dir, exist_ok=True)

                    # --- Создаем бэкап старой модели перед заменой (Quest 8) ---
                    if os.path.exists(settings.MODEL_PATH):
                        backup_filename = f"lgbm_BTCUSDT_1h_backup_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}.pkl"
                        backup_path = os.path.join(os_dir, backup_filename)
                        try:
                            shutil.copy(settings.MODEL_PATH, backup_path)
                            logger.info(
                                f"[Retrain] Успешно создан бэкап старой стабильной модели: {backup_path}"
                            )
                        except Exception as copy_err:
                            logger.error(f"Не удалось скопировать бэкап: {copy_err}")

                    shutil.copy(lgbm_result["model_path"], settings.MODEL_PATH)

                    msg = (
                        f"✅ [Retrain v{version}] Модель обновлена в продакшне.\n"
                        f"accuracy={lgbm_result['metrics']['accuracy']:.3f}, "
                        f"f1={new_f1:.3f} (baseline f1={baseline_f1:.3f})\n"
                    )
                    logger.info(msg)

                    # --- АВТОМАТИЧЕСКАЯ КАЛИБРОВКА (Способ 2) ---
                    try:
                        from scripts.calibrate import get_best_calibration

                        best_sl, best_tp, cal_report = await get_best_calibration(
                            "BTC/USDT", "1h"
                        )

                        # Обновляем настройки в оперативной памяти прямо "на лету"
                        settings.PAPER_SL_PCT = best_sl
                        settings.PAPER_TP_PCT = best_tp

                        msg += f"\n{cal_report}"
                        logger.info(
                            f"[Retrain v{version}] Автокалибровка завершена: SL={best_sl:.1%}, TP={best_tp:.1%}"
                        )
                    except Exception as cal_err:
                        logger.error(f"Ошибка автокалибровки: {cal_err}")
                        msg += f"\n\n⚠️ Автокалибровка завершилась с ошибкой: {cal_err}"

                for admin_id in settings.ADMIN_IDS:
                    try:
                        await bot.send_message(chat_id=admin_id, text=msg)
                    except Exception as e:
                        logger.error(f"Не удалось уведомить админа {admin_id}: {e}")

        except Exception as e:
            logger.error(f"Ошибка в цикле автообучения: {e}")
            await asyncio.sleep(60)


async def on_startup(bot: Bot):
    bot_info = await bot.get_me()
    logger.info(
        f"Bot started: {bot_info.full_name} (@{bot_info.username}, id={bot_info.id})"
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Запускаем фоновую службу бумажной торговли
    asyncio.create_task(paper_trading_loop(bot))
    asyncio.create_task(retrain_loop(bot))


async def on_shutdown(bot: Bot):
    global nexus
    logger.info("Shutting down...")

    if NEXUS_AVAILABLE and nexus:
        await nexus.close()
        logger.info("Nexus SRE сессия успешно завершена.")

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

    # Инициализируем Nexus SRE, если секрет приложения задан
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
