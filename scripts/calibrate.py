"""
Скрипт бэктест-калибровки параметров SL/TP и горизонта удержания на исторических данных БД.
Использование в терминале:
    python -m scripts.calibrate --symbol BTC/USDT --timeframe 1h
"""
import argparse
import os
import asyncio
import pickle
import pandas as pd
from loguru import logger

from src.core.db import AsyncSessionFactory
from src.core.config import get_settings
from src.crud.kline import KlineRepository
from src.features.engineering import add_features
from src.strategy.signals import simulate_strategy
from src.utils.artifact_paths import get_oos_path


def perform_grid_search(
    df_valid: pd.DataFrame,
    sl_grid: list[float] | None = None,
    tp_grid: list[float] | None = None,
    horizon_grid: list[int] = None,
    min_trades: int = 10,
    k_sl_grid: list[float] | None = None,
    k_tp_grid: list[float] | None = None,
) -> list[dict]:
    results = []
    use_atr_mode = k_sl_grid is not None and k_tp_grid is not None

    if use_atr_mode:
        for k_sl in k_sl_grid:
            for k_tp in k_tp_grid:
                for hz in horizon_grid:
                    metrics = simulate_strategy(
                        df_valid, predicted_col="predicted_signal",
                        horizon=hz, sl_pct=None, tp_pct=None,
                        sl_atr_mult=k_sl, tp_atr_mult=k_tp,
                    )
                    if metrics["total_trades"] >= min_trades:
                        results.append({
                            "sl_atr_mult": k_sl,
                            "tp_atr_mult": k_tp,
                            "horizon": hz,
                            "total_trades": metrics["total_trades"],
                            "win_rate": metrics["win_rate"],
                            "profit_factor": metrics["profit_factor"],
                            "sharpe_ratio": metrics["sharpe_ratio"],
                            "sortino_ratio": metrics["sortino_ratio"],
                            "expectancy": metrics["expectancy"],
                            "total_return": metrics["total_return"],
                        })
        return results

    for sl in sl_grid:
        for tp in tp_grid:
            for hz in horizon_grid:
                metrics = simulate_strategy(
                    df_valid, predicted_col="predicted_signal",
                    horizon=hz, sl_pct=sl, tp_pct=tp,
                )
                if metrics["total_trades"] >= min_trades:
                    results.append({
                        "sl_pct": sl,
                        "tp_pct": tp,
                        "horizon": hz,
                        "total_trades": metrics["total_trades"],
                        "win_rate": metrics["win_rate"],
                        "profit_factor": metrics["profit_factor"],
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "sortino_ratio": metrics["sortino_ratio"],
                        "expectancy": metrics["expectancy"],
                        "total_return": metrics["total_return"],
                    })
    return results


async def get_best_calibration(
    symbol: str,
    timeframe: str,
    custom_model_path: str = None,
    use_atr_calibration: bool = False,
) -> tuple[float, float, int, str, dict]:
    """
    Проводит калибровку рисков и горизонта, возвращает: (best_sl, best_tp, best_horizon, formatted_report_text).
    Использует чистые OOS-данные из файла модели для исключения утечек.
    """
    settings = get_settings()
    model_path = custom_model_path if custom_model_path is not None else settings.get_model_path(symbol, timeframe)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели {model_path} не найден.")

    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)

    # 1. Пробуем получить чистые OOS-данные из отдельного parquet-файла (текущий формат)
    oos_path = get_oos_path(model_path)
    df_valid = None

    if os.path.exists(oos_path):
        df_valid = await asyncio.to_thread(pd.read_parquet, oos_path)
        logger.info(
            f"Используются чистые Out-of-Sample данные ({len(df_valid)} строк) из {os.path.basename(oos_path)}"
        )
    else:
        # Обратная совместимость со старыми pkl-артефактами, где df_oos
        # хранился внутри самого pickle. Актуально до тех пор,
        # пока модель не пройдёт очередной цикл переобучения.
        df_valid = saved_data.get("df_oos")
        if df_valid is not None:
            logger.info(
                f"Используются чистые Out-of-Sample данные ({len(df_valid)} строк) "
                f"из легаси-поля df_oos внутри {os.path.basename(model_path)}"
            )

    if df_valid is None:
        logger.warning("Чистые OOS-данные не найдены ни в parquet, ни в pickle. Переход на резервный in-sample метод...")
        async with AsyncSessionFactory() as session:
            repo = KlineRepository(session)
            klines = await repo.get_klines(symbol, timeframe, limit=10000)

        if len(klines) < 100:
            raise ValueError(f"Недостаточно исторических данных в БД ({len(klines)} свечей).")

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
        df = pd.DataFrame(data).sort_values("open_time").reset_index(drop=True)

        df_feats = add_features(df)

        model = saved_data["model"]
        features = saved_data["features"]
        scaler = saved_data.get("scaler")
        target_col = saved_data.get("target_col", "target_binary")

        df_valid = df_feats.dropna(subset=features).copy()
        if df_valid.empty:
            raise ValueError("После расчета признаков не осталось валидных данных.")

        X = df_valid[features]
        if scaler is not None:
            X = scaler.transform(X)

        raw_pred = model.predict(X)
        if target_col == "target_triple":
            signal_map = {0: -1.0, 1: 0.0, 2: 1.0}
            df_valid["predicted_signal"] = pd.Series(
                raw_pred, index=df_valid.index
            ).map(signal_map)
        else:
            df_valid["predicted_signal"] = raw_pred

    if "open_time" in df_valid.columns:
        df_valid = df_valid.sort_values("open_time").reset_index(drop=True)

    calib_split_idx = int(len(df_valid) * 0.7)
    df_calib = df_valid.iloc[:calib_split_idx].reset_index(drop=True)
    df_eval = df_valid.iloc[calib_split_idx:].reset_index(drop=True)

    logger.info(
        f"[Calibration] Разделение OOS: calibration={len(df_calib)} строк, "
        f"honest_eval={len(df_eval)} строк"
    )

    if use_atr_calibration and "atr" in df_calib.columns:
        k_sl_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        k_tp_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        horizon_grid = [3, 5, 8, 12]

        results = await asyncio.to_thread(
            perform_grid_search,
            df_calib,
            k_sl_grid=k_sl_grid,
            k_tp_grid=k_tp_grid,
            horizon_grid=horizon_grid,
            min_trades=settings.CALIBRATION_MIN_TRADES,
        )

        if not results:
            raise ValueError(
                f"Ни одна ATR-комбинация параметров не набрала {settings.CALIBRATION_MIN_TRADES}+ сделок."
            )

        res_df = (
            pd.DataFrame(results)
            .sort_values(by=["sharpe_ratio", "expectancy"], ascending=False)
            .reset_index(drop=True)
        )
        best = res_df.iloc[0]

        honest_metrics = await asyncio.to_thread(
            simulate_strategy,
            df_eval,
            "predicted_signal",
            int(best["horizon"]),
            None,
            None,
            0.001,
            float(best["sl_atr_mult"]),
            float(best["tp_atr_mult"]),
        )

        if honest_metrics["total_trades"] < settings.CALIBRATION_MIN_TRADES:
            logger.warning(
                f"[Calibration] Honest eval split дал только {honest_metrics['total_trades']} "
                f"сделок (< {settings.CALIBRATION_MIN_TRADES}) — оценка может быть шумной."
            )

        report = (
            f"⚙️ <b>Результаты автокалибровки рисков (ATR-режим):</b>\n"
            f"📉 Stop-Loss: <code>{best['sl_atr_mult']:.2f} × ATR</code>\n"
            f"📈 Take-Profit: <code>{best['tp_atr_mult']:.2f} × ATR</code>\n"
            f"⏱ Горизонт (Horizon): <code>{int(best['horizon'])} свечей</code>\n"
            f"📊 Sharpe (grid search, calibration): <code>{best['sharpe_ratio']:.3f}</code>\n"
            f"📊 Sharpe (honest, held-out): <code>{honest_metrics['sharpe_ratio']:.3f}</code>\n"
            f"🎯 Матожидание (honest): <code>{honest_metrics['expectancy']:.3%}</code> на сделку\n"
            f"💰 Доходность (honest): <code>{honest_metrics['total_return']:.1%}</code> "
            f"({honest_metrics['total_trades']} сделок)"
        )

        # ВАЖНО: в ATR-режиме первые два элемента кортежа — это множители ATR,
        # а не проценты. Вызывающий код должен явно это учитывать (см. докстринг).
        return (
            float(best["sl_atr_mult"]),
            float(best["tp_atr_mult"]),
            int(best["horizon"]),
            report,
            honest_metrics,
        )

        # --- существующая фикс.-процентная ветка, без изменений ---
    sl_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    tp_grid = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    horizon_grid = [3, 5, 8, 12]

    # Запускаем тяжелую симуляцию сделок бэктеста в потоке
    results = await asyncio.to_thread(
        perform_grid_search,
        df_calib,
        sl_grid,
        tp_grid,
        horizon_grid=horizon_grid,
        min_trades=settings.CALIBRATION_MIN_TRADES,
    )

    if not results:
        raise ValueError(
            f"Ни одна комбинация параметров не набрала {settings.CALIBRATION_MIN_TRADES}+ сделок."
        )

    res_df = pd.DataFrame(results)

    # Сортировка по Sharpe Ratio, а затем по Expectancy
    res_df = res_df.sort_values(
        by=["sharpe_ratio", "expectancy"], ascending=False
    ).reset_index(drop=True)

    best = res_df.iloc[0]

    # Честная оценка выбранных параметров на данных, не участвовавших в подборе.
    honest_metrics = await asyncio.to_thread(
        simulate_strategy,
        df_eval,
        "predicted_signal",
        int(best["horizon"]),
        float(best["sl_pct"]),
        float(best["tp_pct"]),
    )

    if honest_metrics["total_trades"] < settings.CALIBRATION_MIN_TRADES:
        logger.warning(
            f"[Calibration] Honest eval split дал только {honest_metrics['total_trades']} "
            f"сделок (< {settings.CALIBRATION_MIN_TRADES}) — оценка может быть шумной."
        )

    report = (
        f"⚙️ <b>Результаты автокалибровки рисков:</b>\n"
        f"📉 Stop-Loss (SL): <code>{best['sl_pct']:.1%}</code>\n"
        f"📈 Take-Profit (TP): <code>{best['tp_pct']:.1%}</code>\n"
        f"⏱ Горизонт (Horizon): <code>{int(best['horizon'])} свечей</code>\n"
        f"📊 Sharpe (grid search, calibration): <code>{best['sharpe_ratio']:.3f}</code>\n"
        f"📊 Sharpe (honest, held-out): <code>{honest_metrics['sharpe_ratio']:.3f}</code>\n"
        f"🎯 Матожидание (honest): <code>{honest_metrics['expectancy']:.3%}</code> на сделку\n"
        f"💰 Доходность (honest): <code>{honest_metrics['total_return']:.1%}</code> "
        f"({honest_metrics['total_trades']} сделок)"
    )

    return float(best["sl_pct"]), float(best["tp_pct"]), int(best["horizon"]), report, honest_metrics


async def run_calibration_cli(symbol: str, timeframe: str):
    """Обертка для запуска из консоли."""
    try:
        sl, tp, hz, report, _honest_metrics = await get_best_calibration(symbol, timeframe)
        # Очистим HTML-теги для красивого вывода в терминал
        clean_report = report.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
        print("\n" + "="*60)
        print(clean_report)
        print("="*60)
        print(f"\nДля сохранения параметров между перезапусками пропишите в .env:\n"
              f"PAPER_SL_PCT={sl}\n"
              f"PAPER_TP_PCT={tp}\n"
              f"PAPER_HORIZON={hz}")
    except Exception as e:
        print(f"[-] Ошибка при калибровке: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    asyncio.run(run_calibration_cli(args.symbol, args.timeframe))