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
    автоматически откатывает рабочую модель на последний успешный бэкап-артефакт.
    """
    import glob
    import pickle
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
                f"⚠️ <b>Откат невозможен:</b> файлы бэкапов не найдены в папке {os_dir}!"
            )
            logger.critical(msg)
            for admin_id in settings.ADMIN_IDS:
                try:
                    await bot.send_message(chat_id=admin_id, text=msg)
                except Exception as e:
                    logger.error(f"Не удалось отправить алерт админу {admin_id}: {e}")
            return

        # Сортируем бэкапы от самых новых к самым старым
        backup_files.sort(key=os.path.getmtime, reverse=True)

        # Ищем первый целостный, неповрежденный бэкап-артефакт
        best_backup = None
        backup_artifact = None

        for bf in backup_files:
            try:
                with open(bf, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "model" in data:
                    best_backup = bf
                    backup_artifact = data
                    break
            except Exception as parse_err:
                logger.error(
                    f"[SRE] Файл бэкапа {os.path.basename(bf)} поврежден или несовместим: {parse_err}"
                )
                continue

        if not best_backup or not backup_artifact:
            msg = (
                f"🚨 [CRITICAL SRE] Живые показатели по {symbol} деградировали!\n"
                f"⚠️ <b>Откат невозможен:</b> не найдено ни одного валидного/целостного бэкапа в {os_dir}!"
            )
            logger.critical(msg)
            for admin_id in settings.ADMIN_IDS:
                try:
                    await bot.send_message(chat_id=admin_id, text=msg)
                except Exception as e:
                    logger.error(f"Не удалось отправить алерт админу {admin_id}: {e}")
            return

        try:
            # Выполняем физическую замену рабочей модели на верифицированный бэкап
            shutil.copy(best_backup, model_path)

            try:
                # Включаем кулдаун проверок на 24 часа для сбора новой статистики
                await redis.setex(
                    f"model_rollback_cooldown:{symbol}:{timeframe}", 86400, "1"
                )
            except Exception as re_err:
                logger.error(
                    f"Не удалось записать кулдаун SRE-отката для {symbol}: {re_err}"
                )

            # Извлекаем метаданные из восстановленного артефакта
            restored_model_id = backup_artifact.get(
                "model_id", os.path.basename(best_backup)
            )
            restored_cal = backup_artifact.get("calibration", {})
            restored_sl = restored_cal.get("sl_pct", settings.PAPER_SL_PCT)
            restored_tp = restored_cal.get("tp_pct", settings.PAPER_TP_PCT)

            msg = (
                f"🚨 [CRITICAL SRE] Живые показатели по {symbol} ДЕГРАДИРОВАЛИ!\n"
                f"Win Rate: <code>{win_rate:.1%}</code> (порог: {settings.ROLLBACK_WIN_RATE_THRESHOLD:.1%})\n"
                f"Max Drawdown: <code>{max_dd:.1%}</code> (порог: {settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD:.1%})\n\n"
                f"🔄 <b>Автоматический откат успешно выполнен по {symbol}!</b>\n"
                f"Продакшн-модель заменена на стабильный бэкап-артефакт.\n\n"
                f"🆔 ID восстановленной модели: <code>{restored_model_id}</code>\n"
                f"📉 Восстановленный Stop-Loss: <code>{restored_sl:.1%}</code>\n"
                f"📈 Восстановленный Take-Profit: <code>{restored_tp:.1%}</code>\n\n"
                f"⏱ SRE-проверки по паре заморожены на 24 часа для накопления истории."
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


async def _run_retrain_cycle(bot: Bot, symbol: str, timeframe: str) -> None:
    """
    Выполняет один цикл переобучения для конкретного актива:
    сборка датасета -> обучение baseline -> обучение LGBM с Quality Gate ->
    автокалибровка -> детекция дрейфа -> продвижение в продакшн (или отказ).

    Вынесена из retrain_loop в отдельную функцию, чтобы её можно было
    покрыть юнит-тестами без бесконечного цикла и asyncio.sleep.
    """
    from src.core.db import AsyncSessionFactory
    from src.crud.kline import KlineRepository
    from src.datasets.build import build_and_save_dataset
    from src.models.baseline import run_baseline_experiment
    from src.models.train import run_lgbm_experiment

    settings = get_settings()
    model_path = settings.get_model_path(symbol, timeframe)

    async with AsyncSessionFactory() as session:
        kline_repo = KlineRepository(session)
        klines = await kline_repo.get_klines(
            symbol, timeframe, limit=settings.MIN_KLINES_FOR_TRAIN
        )
        if len(klines) < settings.MIN_KLINES_FOR_TRAIN:
            logger.info(
                f"[Retrain - {symbol}] Недостаточно данных: {len(klines)}/{settings.MIN_KLINES_FOR_TRAIN}"
            )
            return

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
        baseline_f1 = baseline_result["metrics"]["f1"]

        lgbm_result = await run_lgbm_experiment(
            session,
            parquet_path,
            json_path,
            train_size=settings.TRAIN_SIZE,
            test_size=settings.TEST_SIZE,
            models_dir="models/staging",
            baseline_f1=baseline_f1,
        )
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

            # Инициализируем базовое сообщение
            msg = (
                f"✅ [Retrain v{version} - {symbol}] Модель обновлена в продакшне.\n"
                f"accuracy={lgbm_result['metrics']['accuracy']:.3f}, "
                f"f1={new_f1:.3f} (baseline f1={baseline_f1:.3f})\n"
            )

            # --- АВТОМАТИЧЕСКАЯ КАЛИБРОВКА И ВШИВАНИЕ РИСКОВ В АРТЕФАКТ ---
            try:
                from scripts.calibrate import get_best_calibration

                best_sl, best_tp, cal_report = await get_best_calibration(
                    symbol, timeframe, custom_model_path=lgbm_result["model_path"],
                )

                # Открываем артефакт в staging, обновляем параметры калибровки
                with open(lgbm_result["model_path"], "rb") as f:
                    artifact = pickle.load(f)

                artifact["calibration"] = {
                    "sl_pct": best_sl,
                    "tp_pct": best_tp,
                    "calibrated_at": datetime.now(timezone.utc).isoformat(),
                }

                with open(lgbm_result["model_path"], "wb") as f:
                    pickle.dump(artifact, f)

                # Добавляем отчет калибровки к Telegram-сообщению
                msg += f"\n{cal_report}"
                logger.info(
                    f"[Retrain v{version} - {symbol}] Автокалибровка завершена. SL={best_sl:.1%}, TP={best_tp:.1%}"
                )

                # --- АВТОМАТИЧЕСКАЯ ДЕТЕКЦИЯ ДРЕЙФА ПРИЗНАКОВ ---
                try:
                    if os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            old_artifact = pickle.load(f)

                        df_old_oos = old_artifact.get("df_oos")
                        if df_old_oos is not None:
                            from src.features.drift import ConceptDriftDetector

                            df_new = pd.read_parquet(parquet_path)
                            old_features = old_artifact.get("features", [])

                            drift_report = ConceptDriftDetector.detect_drift(
                                reference_df=df_old_oos,
                                current_df=df_new,
                                features=old_features,
                            )

                            if drift_report["drift_detected"]:
                                drift_warning = (
                                    "📊 <b>[SRE] Обнаружен дрейф распределения признаков!</b> "
                                    "Рыночный цикл меняется."
                                )
                                logger.warning(f"[SRE] Concept drift detected for {symbol}")
                                msg += f"\n\n{drift_warning}"
                except Exception as drift_err:
                    logger.error(f"Не удалось выполнить проверку дрейфа признаков: {drift_err}")

            except Exception as cal_err:
                logger.error(f"Ошибка автокалибровки для {symbol}: {cal_err}")
                msg += f"\n\n⚠️ Автокалибровка завершилась с ошибкой: {cal_err}"

            # --- Бэкап старой продакшн-модели перед заменой ---
            if os.path.exists(model_path):
                clean_symbol = symbol.replace("/", "").replace(":", "")
                clean_tf = timeframe.replace("/", "")
                backup_filename = (
                    f"lgbm_{clean_symbol}_{clean_tf}_backup_"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}.pkl"
                )
                backup_path = os.path.join(os_dir, backup_filename)
                try:
                    shutil.copy(model_path, backup_path)
                    logger.info(
                        f"[Retrain - {symbol}] Успешно создан бэкап старой стабильной модели: {backup_path}"
                    )
                except Exception as copy_err:
                    logger.error(f"Не удалось скопировать бэкап для {symbol}: {copy_err}")

            # Финально копируем упакованный и откалиброванный артефакт в продакшн
            shutil.copy(lgbm_result["model_path"], model_path)
            logger.info(f"[Retrain - {symbol}] Новая модель успешно скопирована в продакшн: {model_path}")

        for admin_id in settings.ADMIN_IDS:
            try:
                await bot.send_message(chat_id=admin_id, text=msg)
            except Exception as e:
                logger.error(f"Не удалось уведомить админа {admin_id}: {e}")


async def retrain_loop(bot: Bot, symbol: str, timeframe: str):
    """
    Фоновая служба периодического переобучения модели для конкретного актива (Quest 9).
    """
    settings = get_settings()
    logger.info(f"Фоновая служба автообучения для {symbol} ({timeframe}) запущена.")

    while True:
        try:
            await _run_retrain_cycle(bot, symbol, timeframe)
            await asyncio.sleep(settings.RETRAIN_INTERVAL_SECONDS)

        except Exception as e:
            logger.error(f"Ошибка в цикле автообучения для {symbol}: {e}")
            await asyncio.sleep(60)


async def on_startup(bot: Bot):
    bot_info = await bot.get_me()
    logger.info(
        f"Bot started: {bot_info.full_name} (@{bot_info.username}, id={bot_info.id})"
    )

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
