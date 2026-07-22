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
from src.utils.artifact_paths import get_oos_path
from src.telegram.formatter import TradingNotification, format_trading_notification

import warnings
# Подавляем ложные предупреждения scipy-оптимизатора при обучении LogisticRegression
try:
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)
except ImportError:
    pass


def _atomic_copy(src: str, dst: str) -> None:
    """Копия во временный файл + os.replace — исключает чтение частично записанного .pkl."""
    tmp = dst + ".tmp"
    shutil.copy(src, tmp)
    os.replace(tmp, dst)


def _get_backup_timestamp(filepath: str) -> int:
    """
    Извлекает временную метку (YYYYMMDDHHMM) из имени файла бэкапа.
    Пример имени: /path/to/lgbm_BTCUSDT_1h_backup_202607151030.pkl
    Возвращает целое число 202607151030 для надежной хронологической сортировки.
    При ошибках парсинга возвращает 0.
    """
    try:
        base = os.path.basename(filepath)
        name_without_ext, _ = os.path.splitext(base)
        if "_backup_" in name_without_ext:
            parts = name_without_ext.split("_backup_")
            timestamp_str = parts[-1]
            # Оставляем только цифры
            digits = "".join(c for c in timestamp_str if c.isdigit())
            if digits:
                return int(digits)
    except Exception as e:
        logger.error(f"Ошибка при извлечении временной метки из бэкапа {filepath}: {e}")
    return 0


def _rotate_backups(os_dir: str, clean_symbol: str, clean_tf: str, keep_count: int = 5) -> None:
    """
    Удаляет старые файлы бэкапов для конкретного актива,
    сохраняя только последние keep_count штук на основе временной метки в имени.
    """
    import glob
    backup_pattern = os.path.join(
        os_dir, f"lgbm_{clean_symbol}_{clean_tf}_backup_*.pkl"
    )
    backup_files = glob.glob(backup_pattern)

    # Сортируем от свежих к старым
    backup_files.sort(key=_get_backup_timestamp, reverse=True)

    if len(backup_files) > keep_count:
        files_to_remove = backup_files[keep_count:]
        for bf in files_to_remove:
            try:
                os.remove(bf)
                logger.info(f"[SRE Rotation] Старый файл бэкапа удален: {os.path.basename(bf)}")
            except Exception as err:
                logger.error(f"[SRE Rotation] Ошибка удаления старого бэкапа {bf}: {err}")


nexus = None  # Инициализация для предотвращения NameError в on_shutdown
background_tasks: set[asyncio.Task] = set()


def _start_background_task(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task

# Мягкая SRE интеграция
try:
    from nexus_sdk import NexusSDK

    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    logger.warning("NexusSDK не установлен. Запуск без Nexus SRE.")


def _format_calibration_risk(calibration: dict, settings) -> tuple[str, str]:
    """
    Форматирует SL/TP из артефакта калибровки для Telegram, учитывая, что
    калибровка могла быть выполнена в режиме фиксированных процентов или
    в режиме множителей ATR (см. ATR_RISK_MODEL_ENABLED). Без этой развилки
    множитель вроде 1.5 отобразился бы как "150.0%".
    """
    risk_mode = calibration.get("risk_mode")
    if risk_mode == "atr" and "sl_atr_mult" in calibration and "tp_atr_mult" in calibration:
        return f"{calibration['sl_atr_mult']:.2f} × ATR", f"{calibration['tp_atr_mult']:.2f} × ATR"

    sl_pct = calibration.get("sl_pct", settings.PAPER_SL_PCT)
    tp_pct = calibration.get("tp_pct", settings.PAPER_TP_PCT)
    return f"{sl_pct:.1%}", f"{tp_pct:.1%}"


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
    from src.crud.paper import TradeRepository
    from src.strategy.signals import calculate_strategy_metrics

    settings = get_settings()
    model_path = settings.get_model_path(symbol, timeframe)

    redis = None  # Инициализируем для предотвращения UnboundLocalError / NameError при сбое get_redis()

    # 1. Проверяем кулдаун отката по этой паре в Redis
    try:
        redis = await get_redis()
        cooldown = await redis.get(f"model_rollback_cooldown:{symbol}:{timeframe}")
        if cooldown:
            return
    except Exception as re_err:
        logger.error(f"Ошибка проверки кулдауна в Redis для {symbol}: {re_err}")

    repo = TradeRepository(session)
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

        # Надежная хронологическая сортировка бэкапов от самых новых к самым старым по временной метке в имени
        backup_files.sort(key=_get_backup_timestamp, reverse=True)

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
            _atomic_copy(best_backup, model_path)

            try:
                # Включаем кулдаун проверок на 24 часа для сбора новой статистики
                if redis is not None:
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
            restored_sl_str, restored_tp_str = _format_calibration_risk(
                restored_cal, settings
            )

            msg = (
                f"🚨 [CRITICAL SRE] Живые показатели по {symbol} ДЕГРАДИРОВАЛИ!\n"
                f"Win Rate: <code>{win_rate:.1%}</code> (порог: {settings.ROLLBACK_WIN_RATE_THRESHOLD:.1%})\n"
                f"Max Drawdown: <code>{max_dd:.1%}</code> (порог: {settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD:.1%})\n\n"
                f"🔄 <b>Автоматический откат успешно выполнен по {symbol}!</b>\n"
                f"Продакшн-модель заменена на стабильный бэкап-артефакт.\n\n"
                f"🆔 ID восстановленной модели: <code>{restored_model_id}</code>\n"
                f"📉 Восстановленный Stop-Loss: <code>{restored_sl_str}</code>\n"
                f"📈 Восстановленный Take-Profit: <code>{restored_tp_str}</code>\n\n"
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
    Фоновая служба торговли.
    Автоматически переключается на реальный API Binance при наличии ключей.
    Соблюдает блокировки Kill Switch и сверяет позиции перед каждым раундом.
    """
    from src.data.collector import DataCollector
    from src.models.predictor import Predictor
    from src.crud.kline import KlineRepository

    # Новые импорты SRE контура
    from src.risk.engine import RiskEngine
    from src.risk.kill_switch import KillSwitchManager

    settings = get_settings()
    model_path = settings.get_model_path(symbol, timeframe)
    logger.info(f"Фоновая служба торговли для {symbol} ({timeframe}) запущена.")

    while True:
        try:
            await asyncio.sleep(3600)

            async with AsyncSessionFactory() as session:
                # 1. Скачиваем свечи (всегда полезно иметь актуальную историю для ML)
                if settings.LIVE_TRADING:
                    async with DataCollector(session) as collector:
                        await collector.fetch_and_save_klines(symbol, timeframe, limit=5)

                # 2. Инициализируем SRE менеджеры и проверяем блокировку перед обращением к бирже
                redis = await get_redis()
                kill_switch = KillSwitchManager(redis)

                if await kill_switch.is_trading_blocked():
                    logger.debug(f"[paper_trading_loop] Торговля по {symbol} заблокирована (Kill Switch). Пропускаем раунд.")
                    continue

                risk_engine = RiskEngine()

                # Безальтернативная инициализация коннектора Binance фьючерсов
                exchange = None
                if settings.LIVE_TRADING:
                    from src.exchange.binance import BinanceExchange
                    exchange = BinanceExchange(
                        api_key=settings.BINANCE_API_KEY,
                        secret=settings.BINANCE_API_SECRET,
                        testnet=settings.BINANCE_TESTNET,
                    )

                from src.exchange.engine import TradingEngine

                trading_engine = TradingEngine(
                    exchange=exchange,
                    risk_engine=risk_engine,
                    kill_switch_manager=kill_switch,
                    session=session,
                    settings=settings,
                )

                try:
                    # Reconciliation работает централизованно отдельной фоновой задачей.
                    if await kill_switch.is_trading_blocked():
                        continue

                    # 5. Загружаем свежие данные для предиктора
                    kline_repo = KlineRepository(session)
                    klines = await kline_repo.get_klines(symbol, timeframe, limit=50)
                    data = [
                        {
                            "open_time": k.open_time,
                            "open": k.open,
                            "high": k.high,
                            "low": k.low,
                            "close": k.close,
                            "volume": k.volume,
                        }
                        for k in klines
                    ]
                    df = (
                        pd.DataFrame(data)
                        .sort_values("open_time")
                        .reset_index(drop=True)
                    )

                    predictor = Predictor(model_path)
                    signal = predictor.predict(df)
                    latest_close = df["close"].iloc[-1]

                    # 6. Запускаем торговый движок (создан выше, в шаге 2.5)
                    log_msg = await trading_engine.process_signal(
                        symbol,
                        signal,
                        latest_close,
                        model_id=predictor.model_id,
                        idempotency_key=(
                            f"{predictor.model_id}:{symbol}:{timeframe}:"
                            f"{int(df['open_time'].iloc[-1])}:{signal}"
                        ),
                    )

                    if log_msg:
                        # TradingEngine emits a contract; Telegram owns the text.
                        if isinstance(log_msg, TradingNotification):
                            log_msg = format_trading_notification(log_msg)
                        # Оповещаем администраторов о всех событиях движка
                        for admin_id in settings.ADMIN_IDS:
                            try:
                                await bot.send_message(chat_id=admin_id, text=log_msg)
                            except Exception as e:
                                logger.error(f"Не удалось отправить лог админу: {e}")

                finally:
                    # Закрываем асинхронную сессию CCXT
                    if exchange is not None and hasattr(exchange, "close"):
                        await exchange.close()

        except Exception as e:
            logger.error(f"Ошибка в цикле торговли для {symbol}: {e}")
            await asyncio.sleep(60)


async def account_reconciliation_loop(bot: Bot) -> None:
    """Continuously projects Binance account state into the local ledger."""
    settings = get_settings()
    if not settings.LIVE_TRADING:
        logger.info(
            f"Account reconciliation не запускается в режиме {settings.TRADING_MODE}."
        )
        return

    from src.exchange.binance import BinanceExchange
    from src.risk.kill_switch import KillSwitchManager, KillSwitchState, reconcile_positions

    exchange = BinanceExchange(
        api_key=settings.BINANCE_API_KEY,
        secret=settings.BINANCE_API_SECRET,
        testnet=settings.BINANCE_TESTNET,
    )
    redis = await get_redis()
    kill_switch = KillSwitchManager(redis)
    symbols = [symbol for symbol, _ in settings.ACTIVE_CONFIGS]

    try:
        while True:
            try:
                async with AsyncSessionFactory() as session:
                    synced, details = await reconcile_positions(
                        exchange,
                        session,
                        symbols,
                        kill_switch,
                        environment=settings.TRADING_MODE,
                        verify_protection=True,
                    )
                if not synced:
                    alert_hash = str(hash(details))
                    previous = await redis.get("marketmind:reconciliation:last_alert")
                    if previous != alert_hash:
                        await redis.set(
                            "marketmind:reconciliation:last_alert",
                            alert_hash,
                            ex=300,
                        )
                        message = (
                            "🚨 <b>RECONCILIATION · SAFE MODE</b>\n\n"
                            f"<code>{details}</code>"
                        )
                        for admin_id in settings.ADMIN_IDS:
                            await bot.send_message(chat_id=admin_id, text=message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"Ошибка account reconciliation: {exc}")
            await asyncio.sleep(settings.RECONCILIATION_INTERVAL_SECONDS)
    finally:
        await exchange.close()


async def binance_user_data_loop(bot: Bot) -> None:
    """Primary live-state feed; REST reconciliation remains a control snapshot."""
    settings = get_settings()
    if not settings.LIVE_TRADING:
        return

    from src.crud.execution import ExecutionRepository
    from src.exchange.binance import BinanceExchange
    from src.risk.kill_switch import KillSwitchManager, KillSwitchState, reconcile_positions

    exchange = BinanceExchange(
        api_key=settings.BINANCE_API_KEY,
        secret=settings.BINANCE_API_SECRET,
        testnet=settings.BINANCE_TESTNET,
    )
    redis = await get_redis()
    kill_switch = KillSwitchManager(redis)
    symbols = [symbol for symbol, _ in settings.ACTIVE_CONFIGS]

    try:
        async for event in exchange.user_data_stream():
            try:
                if event.get("e") == "_STREAM_RECONNECTED":
                    async with AsyncSessionFactory() as session:
                        await reconcile_positions(
                            exchange,
                            session,
                            symbols,
                            kill_switch,
                            environment=settings.TRADING_MODE,
                            verify_protection=True,
                        )
                    continue

                async with AsyncSessionFactory() as session:
                    result = await ExecutionRepository(session).apply_user_stream_event(
                        settings.TRADING_MODE, event
                    )
                    if result["order_changed"] or result["symbols"]:
                        # An event is authoritative for fills; the REST call is
                        # a targeted control check for position/protection drift.
                        await reconcile_positions(
                            exchange,
                            session,
                            symbols,
                            kill_switch,
                            environment=settings.TRADING_MODE,
                            verify_protection=True,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"Ошибка обработки Binance User Data Stream: {exc}")
                await kill_switch.set_state(
                    KillSwitchState.SAFE_MODE,
                    "USER_STREAM_EVENT_FAILED",
                    str(exc),
                )
    finally:
        await exchange.close()


_TRAIN_SEMAPHORE = asyncio.Semaphore(1)

async def _run_retrain_cycle(bot: Bot, symbol: str, timeframe: str) -> None:
    from src.core.db import AsyncSessionFactory
    from src.crud.kline import KlineRepository
    from src.datasets.build import build_and_save_dataset
    from src.models.baseline import run_baseline_experiment, compute_baseline_holdout_f1
    from src.models.train import run_lgbm_experiment

    settings = get_settings()
    model_path = settings.get_model_path(symbol, timeframe)

    async with AsyncSessionFactory() as session:
        kline_repo = KlineRepository(session)
        klines = await kline_repo.get_klines(
            symbol, timeframe, limit=settings.MIN_KLINES_FOR_TRAIN
        )
        if len(klines) < settings.MIN_KLINES_FOR_TRAIN:
            logger.debug(
                f"[Retrain - {symbol}] Недостаточно данных для переобучения: {len(klines)}/{settings.MIN_KLINES_FOR_TRAIN}"
            )
            return

        # Дешёвая проверка (чтение из БД) сделана вне семафора — держать
        # блокировку ради неё незачем. Всё, что тяжело по памяти, — под ней.
        async with _TRAIN_SEMAPHORE:
            version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")

            parquet_path = await build_and_save_dataset(
                session,
                symbol=symbol,
                timeframe=timeframe,
                version=version,
                horizon=settings.LABEL_HORIZON,
                threshold=settings.LABEL_THRESHOLD,
                tp_atr_mult=settings.LABEL_TP_ATR_MULT
                if settings.ATR_RISK_MODEL_ENABLED
                else None,
                sl_atr_mult=settings.LABEL_SL_ATR_MULT
                if settings.ATR_RISK_MODEL_ENABLED
                else None,
            )
            json_path = parquet_path.replace(".parquet", ".json")

            baseline_result = await run_baseline_experiment(
                session,
                parquet_path,
                json_path,
                train_size=settings.TRAIN_SIZE,
                test_size=settings.TEST_SIZE,
            )
            # baseline_f1 (усредненный по всей истории Walk-Forward) используется
            # ниже только для сравнения new_f1 vs baseline_f1 в отчете о продвижении
            # в прод — там обе метрики одной природы (усреднение по фолдам).
            baseline_f1 = baseline_result["metrics"]["f1"]

            # Для внутреннего Quality Gate в run_lgbm_experiment нужен baseline,
            # честно сравнимый с holdout_f1 LGBM — то есть посчитанный на ТОМ ЖЕ
            # train_val/holdout split, а не усредненный по всей истории.
            baseline_holdout_result = await compute_baseline_holdout_f1(
                session,
                parquet_path,
                json_path,
                train_size=settings.TRAIN_SIZE,
                test_size=settings.TEST_SIZE,
            )
            gate_baseline_f1 = baseline_holdout_result["f1"]
            if gate_baseline_f1 is None:
                gate_baseline_f1 = baseline_f1

            regime_drift_pvalue = None
            if os.path.exists(model_path):
                try:
                    with open(model_path, "rb") as f:
                        old_artifact_for_drift = pickle.load(f)
                    old_oos_path = get_oos_path(model_path)
                    if os.path.exists(old_oos_path):
                        from src.features.drift import ConceptDriftDetector

                        df_old_oos_pre = await asyncio.to_thread(
                            pd.read_parquet, old_oos_path
                        )
                        df_new_pre = await asyncio.to_thread(
                            pd.read_parquet, parquet_path
                        )
                        old_features_pre = old_artifact_for_drift.get("features", [])

                        pre_drift_report = await asyncio.to_thread(
                            ConceptDriftDetector.detect_drift,
                            reference_df=df_old_oos_pre,
                            current_df=df_new_pre,
                            features=old_features_pre,
                        )
                        p_values = [
                            r["p_value"]
                            for r in pre_drift_report.get("results", {}).values()
                            if "p_value" in r
                        ]
                        if p_values:
                            regime_drift_pvalue = float(min(p_values))
                except Exception as pre_drift_err:
                    logger.error(
                        f"Не удалось посчитать дрейф до обучения для {symbol}: {pre_drift_err}"
                    )

            try:
                lgbm_result = await run_lgbm_experiment(
                    session,
                    parquet_path,
                    json_path,
                    train_size=settings.TRAIN_SIZE,
                    test_size=settings.TEST_SIZE,
                    models_dir="models/staging",
                    baseline_f1=gate_baseline_f1,
                    regime_drift_pvalue=regime_drift_pvalue,
                )
                new_f1 = lgbm_result["metrics"]["f1"]
            except ValueError as gate_err:
                err_msg = str(gate_err)
                if "REJECTED" in err_msg:
                    logger.warning(f"[Retrain - {symbol}] {err_msg}")
                    reject_msg = (
                        f"⚠️ {symbol} v{version} → <b>ОТКЛОНЕНА</b> (F1 Quality Gate)\n\n"
                        f"🤖 <code>{err_msg}</code>"
                    )
                    for admin_id in settings.ADMIN_IDS:
                        try:
                            await bot.send_message(chat_id=admin_id, text=reject_msg)
                        except Exception as e:
                            logger.error(f"Не удалось отправить алерт админу: {e}")
                    return
                else:
                    raise gate_err

            decision_f1 = lgbm_result["metrics"].get("holdout_f1")
            if decision_f1 is None:
                decision_f1 = new_f1

            # Формируем компактную строчку по классификации ML
            f1_delta = decision_f1 - gate_baseline_f1
            f1_delta_sign = "+" if f1_delta >= 0 else ""
            ml_metrics_block = (
                f"F1 {decision_f1:.3f} (+{f1_delta:.3f} vs BL {gate_baseline_f1:.3f})\n"
                f"Acc {lgbm_result['metrics']['accuracy']:.3f}\n"
                f"CV {new_f1:.3f}"
            )

            if decision_f1 <= gate_baseline_f1:
                msg = (
                    f"⚠️ {symbol} v{version} → <b>ОТКЛОНЕНА</b> (F1 Gate)\n\n"
                    f"{ml_metrics_block}\n"
                    f"Причина: F1 не превысил baseline."
                )
                logger.warning(msg)
            else:
                os_dir = os.path.dirname(model_path)
                if os_dir:
                    os.makedirs(os_dir, exist_ok=True)

                msg = f"{ml_metrics_block}\n"

                artifact = None
                cal_report = ""
                drift_warning = ""

                try:
                    with open(lgbm_result["model_path"], "rb") as f:
                        artifact = pickle.load(f)

                    from scripts.calibrate import get_best_calibration

                    (
                        best_sl,
                        best_tp,
                        best_hz,
                        cal_report,
                        honest_metrics,
                    ) = await get_best_calibration(
                        symbol,
                        timeframe,
                        custom_model_path=lgbm_result["model_path"],
                        meta_model=artifact.get("meta_model"),
                        meta_features=artifact.get("meta_features"),
                        meta_threshold=settings.META_LABELING_THRESHOLD,
                        use_atr_calibration=settings.ATR_RISK_MODEL_ENABLED,
                    )

                    calibration_update = {
                        "horizon": best_hz,
                        "calibrated_at": datetime.now(timezone.utc).isoformat(),
                        "risk_mode": "atr"
                        if settings.ATR_RISK_MODEL_ENABLED
                        else "fixed_pct",
                    }
                    if settings.ATR_RISK_MODEL_ENABLED:
                        calibration_update["sl_atr_mult"] = best_sl
                        calibration_update["tp_atr_mult"] = best_tp
                    else:
                        calibration_update["sl_pct"] = best_sl
                        calibration_update["tp_pct"] = best_tp

                    artifact.setdefault("calibration", {}).update(calibration_update)
                    artifact["backtest_metrics"] = honest_metrics

                    with open(lgbm_result["model_path"], "wb") as f:
                        pickle.dump(artifact, f)

                    logger.info(
                        f"[Retrain v{version} - {symbol}] Автокалибровка завершена. SL={best_sl:.1%}, TP={best_tp:.1%}, Horizon={best_hz}"
                    )

                    try:
                        if os.path.exists(model_path):
                            with open(model_path, "rb") as f:
                                old_artifact = pickle.load(f)

                            old_oos_path = get_oos_path(model_path)
                            if os.path.exists(old_oos_path):
                                from src.features.drift import ConceptDriftDetector

                                df_old_oos = await asyncio.to_thread(
                                    pd.read_parquet, old_oos_path
                                )
                                df_new = await asyncio.to_thread(
                                    pd.read_parquet, parquet_path
                                )
                                old_features = old_artifact.get("features", [])

                                drift_report = await asyncio.to_thread(
                                    ConceptDriftDetector.detect_drift,
                                    reference_df=df_old_oos,
                                    current_df=df_new,
                                    features=old_features,
                                )

                                if drift_report["drift_detected"]:
                                    drift_warning = "📡 Дрейф обнаружен"
                                    logger.warning(f"[SRE] Concept drift detected for {symbol}")
                    except Exception as drift_err:
                        logger.error(f"Не удалось выполнить проверку дрейфа признаков: {drift_err}")

                except Exception as cal_err:
                    logger.exception(f"Ошибка автокалибровки для {symbol}")

                    msg += f"\n⚠️ Автокалибровка завершилась с ошибкой: {type(cal_err).__name__}: {cal_err}"

                    # --- ECONOMIC QUALITY GATE ---
                economic_gate_passed = True
                economic_gate_msg = ""

                new_backtest_metrics = (
                    artifact.get("backtest_metrics") if artifact is not None else None
                )

                if artifact is None:
                    logger.warning(
                        f"[Economic Gate] Калибровка для {symbol} не удалась — метрики "
                        f"прибыльности недоступны, гейт пропущен."
                    )

                if os.path.exists(model_path) and new_backtest_metrics is not None:
                    try:
                        with open(model_path, "rb") as f:
                            current_prod_artifact = pickle.load(f)
                        prod_backtest_metrics = current_prod_artifact.get(
                            "backtest_metrics"
                        )
                    except Exception as read_err:
                        logger.error(
                            f"[Economic Gate] Не удалось прочитать метрики текущей прод-модели {symbol}: {read_err}"
                        )
                        prod_backtest_metrics = None

                    if prod_backtest_metrics is not None:
                        new_sharpe = new_backtest_metrics.get("sharpe_ratio", 0.0)
                        new_expectancy = new_backtest_metrics.get("expectancy", 0.0)
                        prod_sharpe = prod_backtest_metrics.get("sharpe_ratio", 0.0)

                        if new_expectancy <= 0:
                            economic_gate_passed = False
                            economic_gate_msg = f"отрицательное матожидание на честном OOS бэктесте (<code>{new_expectancy:.3%}</code>)"
                        elif new_sharpe <= prod_sharpe:
                            economic_gate_passed = False
                            economic_gate_msg = f"honest Sharpe (<code>{new_sharpe:.3f}</code>) не превышает текущий прод Sharpe (<code>{prod_sharpe:.3f}</code>)"

                # Сборка финального отчета
                if not economic_gate_passed:
                    logger.warning(economic_gate_msg)

                    reject_header = f"⚠️ {symbol} v{version} → <b>ОТКЛОНЕНА</b> (Economic Gate)\n\n"
                    reject_msg = reject_header + msg + "\n"
                    if drift_warning:
                        reject_msg += f"{drift_warning}\n"
                    reject_msg += f"Причина: {economic_gate_msg}."

                    for admin_id in settings.ADMIN_IDS:
                        try:
                            await bot.send_message(chat_id=admin_id, text=reject_msg)
                        except Exception as e:
                            logger.error(f"Не удалось уведомить админа {admin_id}: {e}")
                    return

                # --- END ECONOMIC QUALITY GATE ---

                accept_header = f"✅ {symbol} v{version} → <b>ПРИНЯТА В ПРОД</b>\n\n"
                msg = accept_header + msg + cal_report
                if drift_warning:
                    msg += f"\n{drift_warning}"

                if os.path.exists(model_path):
                    clean_symbol = symbol.replace("/", "").replace(":", "")
                    clean_tf = timeframe.replace("/", "")
                    backup_filename = (
                        f"lgbm_{clean_symbol}_{clean_tf}_backup_"
                        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}.pkl"
                    )
                    backup_path = os.path.join(os_dir, backup_filename)

                    try:
                        _atomic_copy(model_path, backup_path)
                        logger.info(
                            f"[Retrain - {symbol}] Успешно создан бэкап старой стабильной модели: {backup_path}"
                        )
                        _rotate_backups(os_dir, clean_symbol, clean_tf, keep_count=5)

                    except Exception as copy_err:
                        logger.error(
                            f"Не удалось скопировать бэкап для {symbol}: {copy_err}"
                        )

                _atomic_copy(lgbm_result["model_path"], model_path)
                logger.info(
                    f"[Retrain - {symbol}] Новая модель успешно скопирована в продакшн: {model_path}"
                )

                staging_oos_path = get_oos_path(lgbm_result["model_path"])
                if os.path.exists(staging_oos_path):
                    production_oos_path = get_oos_path(model_path)
                    try:
                        _atomic_copy(staging_oos_path, production_oos_path)
                    except Exception as oos_copy_err:
                        logger.error(
                            f"Не удалось скопировать OOS-parquet для {symbol}: {oos_copy_err}"
                        )

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
        _start_background_task(paper_trading_loop(bot, symbol, timeframe))
        _start_background_task(retrain_loop(bot, symbol, timeframe))
    _start_background_task(account_reconciliation_loop(bot))
    _start_background_task(binance_user_data_loop(bot))


async def on_shutdown(bot: Bot):
    global nexus
    logger.info("Shutting down...")

    tasks = list(background_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

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
