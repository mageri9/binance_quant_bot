import asyncio
import sys
from aiogram import Bot, Dispatcher
from loguru import logger

from src.config import get_settings
from src.db import init_db
from src.event_bus import AsyncEventBus
from src.exchange.paper import PaperExchange
from src.exchange.binance import BinanceExchange
from src.services.market import MarketService
from src.services.strategy_service import StrategyService
from src.services.risk_service import RiskService
from src.services.execution_service import ExecutionService
from src.services.notifier_service import NotifierService
from src.bot.handlers import router as bot_router

# Nexus SRE SDK Интеграция
nexus_sdk = None
try:
    from nexus_sdk import NexusSDK
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False


async def main():
    global nexus_sdk
    settings = get_settings()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("Запуск MarketMind Quant Bot...")

    # 1. Инициализация БД & Шины событий
    await init_db()
    bus = AsyncEventBus()

    # 2. Выбор адаптера биржи по флагу TRADING_MODE (paper | testnet | mainnet)
    if settings.TRADING_MODE == "paper":
        exchange = PaperExchange()
        logger.info("[Main] Режим биржи: PAPER (Локальная симуляция)")
    else:
        exchange = BinanceExchange(
            api_key=settings.BINANCE_API_KEY,
            secret=settings.BINANCE_API_SECRET,
            testnet=(settings.TRADING_MODE == "testnet"),
            proxy=settings.BINANCE_PROXY
        )
        logger.info(f"[Main] Режим биржи: BINANCE {settings.TRADING_MODE.upper()}")

    # 3. Настройка Telegram Бота
    bot = Bot(token=settings.BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(bot_router)
    dp["exchange"] = exchange  # прокидываем объект биржи в Telegram-хэндлеры

    # 4. Инициализация Nexus SDK (Ваш SRE мониторинг)
    if NEXUS_AVAILABLE and settings.NEXUS_APP_SECRET:
        nexus_sdk = NexusSDK(
            endpoint_url=settings.NEXUS_ENDPOINT_URL,
            app_secret=settings.NEXUS_APP_SECRET,
            project_name=settings.NEXUS_PROJECT_NAME
        )
        nexus_sdk.register_aiogram_error_handler(dp)
        nexus_sdk.start_heartbeat(interval_seconds=15)
        logger.info("[Main] Nexus SRE SDK успешно подключен.")

    # 5. Инициализация Сервисов-Обработчиков событий
    market_service = MarketService(bus, exchange)
    strategy_service = StrategyService(bus)
    risk_service = RiskService(bus, exchange)
    execution_service = ExecutionService(bus, exchange)
    notifier_service = NotifierService(bus, bot, nexus_sdk)

    # 6. Запуск фонового опроса свечей
    asyncio.create_task(market_service.start_polling(interval_seconds=60))

    logger.info("[Main] Бот запущен и готов к работе!")

    try:
        await dp.start_polling(bot)
    finally:
        if hasattr(exchange, "close"):
            await exchange.close()
        if nexus_sdk:
            await nexus_sdk.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")