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
    meta_model=None,
    meta_features: list[str] | None = None,
    meta_threshold: float | None = None,
) -> tuple[float, float, int, str, dict]:
    """
    Проводит калибровку рисков и горизонта, возвращает: (best_sl, best_tp, best_horizon, formatted_report_text, honest_metrics).

    ВАЖНО: в ATR-режиме (use_atr_calibration=True и колонка 'atr' присутствует)
    возвращаемые best_sl/best_tp — это МНОЖИТЕЛИ ATR, а не проценты.
    Вызывающий код обязан проверять settings.ATR_RISK_MODEL_ENABLED
    и интерпретировать значения соответственно.
    """
    settings = get_settings()
    model_path = custom_model_path if custom_model_path is not None else settings.get_model_path(symbol, timeframe)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели {model_path} не найден.")

    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)

    oos_path = get_oos_path(model_path)
    df_valid = None

    if os.path.exists(oos_path):
        df_valid = await asyncio.to_thread(pd.read_parquet, oos_path)
        logger.info(
            f"Используются чистые Out-of-Sample данные ({len(df_valid)} строк) из {os.path.basename(oos_path)}"
        )
    else:
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

    if meta_model is not None:
        from src.models.meta import apply_meta_gate

        threshold = (
            meta_threshold if meta_threshold is not None else settings.META_LABELING_THRESHOLD
        )
        df_valid["predicted_signal"] = apply_meta_gate(
            df_valid, meta_model, meta_features, threshold,
        )
        logger.info(f"[Calibration] Meta-gate применён к сигналам перед калибровкой ({symbol}).")

    calib_split_idx = int(len(df_valid) * 0.7)
    df_calib = df_valid.iloc[:calib_split_idx].reset_index(drop=True)
    df_eval = df_valid.iloc[calib_split_idx:].reset_index(drop=True)

    logger.info(
        f"[Calibration] Разделение OOS: calibration={len(df_calib)} строк, "
        f"honest_eval={len(df_eval)} строк"
    )

    is_atr_mode = use_atr_calibration and "atr" in df_calib.columns

    if is_atr_mode:
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

        best_sl_value = float(best["sl_atr_mult"])
        best_tp_value = float(best["tp_atr_mult"])

        risk_line = (
            f"🛡️ SL <code>{best_sl_value:.2f}×ATR</code> • "
            f"TP <code>{best_tp_value:.2f}×ATR</code> • "
            f"HZ <code>{int(best['horizon'])}</code>"
        )
    else:
        sl_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        tp_grid = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
        horizon_grid = [3, 5, 8, 12]

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

        res_df = pd.DataFrame(results).sort_values(
            by=["sharpe_ratio", "expectancy"], ascending=False
        ).reset_index(drop=True)
        best = res_df.iloc[0]

        honest_metrics = await asyncio.to_thread(
            simulate_strategy,
            df_eval,
            "predicted_signal",
            int(best["horizon"]),
            float(best["sl_pct"]),
            float(best["tp_pct"]),
        )

        best_sl_value = float(best["sl_pct"])
        best_tp_value = float(best["tp_pct"])


    if honest_metrics["total_trades"] < settings.CALIBRATION_MIN_TRADES:
        logger.warning(
            f"[Calibration] Honest eval split дал только {honest_metrics['total_trades']} "
            f"сделок (< {settings.CALIBRATION_MIN_TRADES}) — оценка может быть шумной."
        )

    if is_atr_mode:
        risk_line = (
            f"SL {best_sl_value:.2f}×ATR\n"
            f"TP {best_tp_value:.2f}×ATR\n"
            f"Горизонт {best['horizon']} свечей"
        )
    else:
        risk_line = (
            f"SL {best_sl_value:.1%}\n"
            f"TP {best_tp_value:.1%}\n"
            f"Горизонт {best['horizon']} свечей"
        )

    report = (
        f"{risk_line}\n\n"
        f"Sharpe {honest_metrics['sharpe_ratio']:.3f}\n"
        f"E[R] {honest_metrics['expectancy']:.3%}\n"
        f"Доходность {honest_metrics['total_return']:.1%} "
        f"({honest_metrics['total_trades']} сд.)"
    )

    return best_sl_value, best_tp_value, int(best["horizon"]), report, honest_metrics


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