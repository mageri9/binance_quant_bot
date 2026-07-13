import asyncio
import sys
import os
import pandas as pd

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


async def on_startup(bot: Bot):
    bot_info = await bot.get_me()
    logger.info(
        f"Bot started: {bot_info.full_name} (@{bot_info.username}, id={bot_info.id})"
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Запускаем фоновую службу бумажной торговли
    asyncio.create_task(paper_trading_loop(bot))


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
