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
    Фоновая служба торговли.
    Автоматически переключается на реальный API Binance при наличии ключей.
    Соблюдает блокировки Kill Switch и сверяет позиции перед каждым раундом.
    """
    from src.data.collector import DataCollector
    from src.models.predictor import Predictor
    from src.crud.kline import KlineRepository

    # Новые импорты SRE контура
    from src.exchange.binance import BinanceExchange
    from src.risk.engine import RiskEngine
    from src.risk.kill_switch import KillSwitchManager, reconcile_positions

    settings = get_settings()
    model_path = settings.get_model_path(symbol, timeframe)
    logger.info(f"Фоновая служба торговли для {symbol} ({timeframe}) запущена.")

    while True:
        try:
            await asyncio.sleep(3600)

            async with AsyncSessionFactory() as session:
                # 1. Скачиваем свечи
                async with DataCollector(session) as collector:
                    await collector.fetch_and_save_klines(symbol, timeframe, limit=5)

                # 2. Инициализируем SRE менеджеры и Биржу
                redis = await get_redis()
                kill_switch = KillSwitchManager(redis)
                risk_engine = RiskEngine()

                # Безальтернативная инициализация коннектора Binance фьючерсов
                exchange = BinanceExchange(
                    api_key=settings.BINANCE_API_KEY,
                    secret=settings.BINANCE_API_SECRET,
                    testnet=(settings.TRADING_MODE == "testnet"),
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
                    # 2.5. Синхронизируем закрытие живых позиций (SL/TP на бирже)
                    # ДО сверки reconcile_positions, иначе легитимное закрытие
                    # по SL/TP будет ложно трактовано как рассинхронизация
                    # и заблокирует бота в SAFE_MODE.
                    close_msg = await trading_engine.check_and_close_positions(symbol)
                    if close_msg:
                        for admin_id in settings.ADMIN_IDS:
                            try:
                                await bot.send_message(chat_id=admin_id, text=close_msg)
                            except Exception as e:
                                logger.error(
                                    f"Не удалось отправить лог о закрытии позиции админу: {e}"
                                )

                    # 3. Сверка позиций перед раундом (Биржа — источник истины)
                    symbols_to_sync = [config[0] for config in settings.ACTIVE_CONFIGS]
                    synced, error_details = await reconcile_positions(
                        exchange, session, symbols_to_sync, kill_switch
                    )

                    if not synced:
                        # error_details уже содержит точные символы всех найденных
                        # mismatch'ей построчно — не привязываем заголовок к локальному
                        # symbol этого цикла, иначе алерт вводит в заблуждение (баг:
                        # три параллельных цикла проверяют один и тот же полный список
                        # пар и репортят чужие детали под своим заголовком).
                        alert_msg = (
                            f"🚨 [SRE RECONCILE ERROR] Обнаружена рассинхронизация позиций!\n"
                            f"Бот заблокирован в SAFE_MODE.\n"
                            f"<code>{error_details}</code>"
                        )
                        for admin_id in settings.ADMIN_IDS:
                            await bot.send_message(chat_id=admin_id, text=alert_msg)
                        continue

                    # 4. Проверяем Kill Switch перед генерацией сигналов
                    if await kill_switch.is_trading_blocked():
                        continue

                    if not os.path.exists(model_path):
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
                        symbol, signal, latest_close
                    )

                    if log_msg:
                        # Оповещаем администраторов о всех событиях движка
                        for admin_id in settings.ADMIN_IDS:
                            try:
                                await bot.send_message(chat_id=admin_id, text=log_msg)
                            except Exception as e:
                                logger.error(f"Не удалось отправить лог админу: {e}")

                finally:
                    # Закрываем асинхронную сессию CCXT
                    if hasattr(exchange, "close"):
                        await exchange.close()

        except Exception as e:
            logger.error(f"Ошибка в цикле торговли для {symbol}: {e}")
            await asyncio.sleep(60)


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
            logger.info(
                f"[Retrain - {symbol}] Недостаточно данных: {len(klines)}/{settings.MIN_KLINES_FOR_TRAIN}"
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

            try:
                lgbm_result = await run_lgbm_experiment(
                    session,
                    parquet_path,
                    json_path,
                    train_size=settings.TRAIN_SIZE,
                    test_size=settings.TEST_SIZE,
                    models_dir="models/staging",
                    baseline_f1=gate_baseline_f1,
                )
                new_f1 = lgbm_result["metrics"]["f1"]
            except ValueError as gate_err:
                err_msg = str(gate_err)
                if "REJECTED" in err_msg:
                    logger.warning(f"[Retrain - {symbol}] {err_msg}")
                    reject_msg = f"⚠️ [Retrain - {symbol}] {err_msg}. В продакшн остается старая модель."
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
                decision_f1 = (
                    new_f1  # fallback для случая без holdout (крошечные датасеты)
                )

            # Решение о продвижении в прод опирается на ту же метрику, что уже
            # прошла Quality Gate внутри run_lgbm_experiment (honest holdout F1),
            # а не на усредненный по walk-forward фолдам F1 — это разные величины
            # с разным составом тестовых окон, их расхождение давало ложные отклонения.
            if decision_f1 <= gate_baseline_f1:
                msg = (
                    f"⚠️ [Retrain v{version} - {symbol}] Новая модель НЕ превзошла baseline "
                    f"(LGBM holdout f1={decision_f1:.3f} vs baseline f1={gate_baseline_f1:.3f}). "
                    f"В продакшн НЕ продвигается."
                )
                logger.warning(msg)
            else:
                os_dir = os.path.dirname(model_path)
                if os_dir:
                    os.makedirs(os_dir, exist_ok=True)

                msg = (
                    f"✅ [Retrain v{version} - {symbol}] Модель обновлена в продакшне.\n"
                    f"accuracy={lgbm_result['metrics']['accuracy']:.3f}, "
                    f"holdout_f1={decision_f1:.3f} (baseline f1={gate_baseline_f1:.3f}), "
                    f"cv_avg_f1={new_f1:.3f}\n"
                )

                artifact = None
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
                    )

                    artifact.setdefault("calibration", {}).update(
                        {
                            "sl_pct": best_sl,
                            "tp_pct": best_tp,
                            "horizon": best_hz,
                            "calibrated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    artifact["backtest_metrics"] = honest_metrics

                    with open(lgbm_result["model_path"], "wb") as f:
                        pickle.dump(artifact, f)

                    msg += f"\n{cal_report}"
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

                    # --- ECONOMIC QUALITY GATE ---
                    # F1-гейт проверяет только качество классификации. Модель может
                    # иметь лучший F1, чем прод, и при этом быть убыточной (плохое R:R,
                    # издержки, слишком широкий/узкий SL/TP). Сравниваем честные
                    # (held-out, не участвовавшие в grid search) метрики прибыльности
                    # кандидата с сохранёнными метриками текущей прод-модели.


                economic_gate_passed = True
                economic_gate_msg = ""

                new_backtest_metrics = (
                    artifact.get("backtest_metrics") if artifact is not None else None
                )

                if artifact is None:
                    logger.warning(
                        f"[Economic Gate] Калибровка для {symbol} не удалась — метрики "
                        f"прибыльности недоступны, гейт пропущен (модель продвигается по F1-критерию)."
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

                    if prod_backtest_metrics is None:
                        logger.warning(
                            f"[Economic Gate] У текущей прод-модели {symbol} нет сохранённых "
                            f"backtest_metrics (задеплоена до внедрения гейта) — сравнение пропущено."
                        )

                    else:
                        new_sharpe = new_backtest_metrics.get("sharpe_ratio", 0.0)
                        new_expectancy = new_backtest_metrics.get("expectancy", 0.0)
                        prod_sharpe = prod_backtest_metrics.get("sharpe_ratio", 0.0)

                        if new_expectancy <= 0:
                            economic_gate_passed = False

                            economic_gate_msg = (
                                f"⚠️ [Economic Gate - {symbol}] Модель ОТКЛОНЕНА: отрицательное "
                                f"матожидание на честном held-out backtest (expectancy={new_expectancy:.3%})."
                            )

                        elif new_sharpe <= prod_sharpe:
                            economic_gate_passed = False

                            economic_gate_msg = (
                                f"⚠️ [Economic Gate - {symbol}] Модель ОТКЛОНЕНА: honest Sharpe "
                                f"({new_sharpe:.3f}) не превышает текущий прод Sharpe ({prod_sharpe:.3f})."
                            )

                if not economic_gate_passed:
                    logger.warning(economic_gate_msg)

                    msg += f"\n\n{economic_gate_msg}\nМодель прошла F1-гейт, но НЕ продвигается в продакшн (Economic Gate)."

                    for admin_id in settings.ADMIN_IDS:
                        try:
                            await bot.send_message(chat_id=admin_id, text=msg)

                        except Exception as e:
                            logger.error(f"Не удалось уведомить админа {admin_id}: {e}")

                    return

                # --- END ECONOMIC QUALITY GATE ---

                msg = f"✅ [Retrain v{version} - {symbol}] Модель обновлена в продакшне.\n" + msg

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