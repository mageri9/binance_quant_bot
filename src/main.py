import asyncio
import sys
import os
import pandas as pd
import shutil

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.redis import RedisStorage
from loguru import logger

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
                    for admin_id in settings.ADMIN_IDS:
                        try:
                            await bot.send_message(chat_id=admin_id, text=log_msg)
                        except Exception as e:
                            logger.error(
                                f"Не удалось отправить уведомление админу {admin_id}: {e}"
                            )

        except Exception as e:
            logger.error(f"Ошибка в цикле бумажной торговли: {e}")
            await asyncio.sleep(60)


nexus = None


async def retrain_loop(bot: Bot):
    """
    Фоновая служба периодического переобучения модели.
    Новая LightGBM-модель продвигается в продакшн только если её f1
    превосходит baseline того же прогона — иначе это сигнал искать
    проблему (leakage, смена режима рынка), а не молча подменять модель.
    """
    from src.core.db import AsyncSessionFactory
    from src.crud.kline import KlineRepository
    from src.datasets.build import build_and_save_dataset
    from src.models.baseline import run_baseline_experiment
    from src.models.train import run_lgbm_experiment
    from datetime import datetime, timezone

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
            # Глобальный перехватчик ошибок в aiogram 3
            nexus.register_aiogram_error_handler(dp)
            # Запуск пульса (Heartbeat) раз в 15 секунд
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
