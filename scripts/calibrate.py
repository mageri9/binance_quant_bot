"""
Скрипт бэктест-калибровки параметров SL/TP на исторических данных БД.
Использование в терминале:
    python -m scripts.calibrate --symbol BTC/USDT --timeframe 1h
"""
import argparse
import asyncio
import os
import pickle
import pandas as pd
from loguru import logger

from src.core.db import AsyncSessionFactory
from src.core.config import get_settings
from src.crud.kline import KlineRepository
from src.features.engineering import add_features
from src.strategy.signals import simulate_strategy


def perform_grid_search(
    df_valid: pd.DataFrame,
    sl_grid: list[float],
    tp_grid: list[float],
    horizon: int,
    min_trades: int = 10,
) -> list[dict]:
    results = []
    for sl in sl_grid:
        for tp in tp_grid:
            metrics = simulate_strategy(
                df_valid, predicted_col="predicted_signal",
                horizon=horizon, sl_pct=sl, tp_pct=tp,
            )
            if metrics["total_trades"] >= min_trades:
                results.append({
                    "sl_pct": sl,
                    "tp_pct": tp,
                    "total_trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "profit_factor": metrics["profit_factor"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "sortino_ratio": metrics["sortino_ratio"],
                    "expectancy": metrics["expectancy"],
                    "total_return": metrics["total_return"],
                })
    return results


async def get_best_calibration(symbol: str, timeframe: str, custom_model_path: str = None) -> tuple[float, float, str]:
    """
    Проводит калибровку и возвращает: (best_sl, best_tp, formatted_report_text).
    Использует чистые OOS-данные из файла модели для исключения утечек.
    """
    settings = get_settings()
    # Если передан custom_model_path, берем его, иначе стандартный из настроек
    model_path = custom_model_path if custom_model_path is not None else settings.get_model_path(symbol, timeframe)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели {model_path} не найден.")

    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)

    # 1. Пробуем получить чистые OOS-данные из файла модели
    df_valid = saved_data.get("df_oos")

    if df_valid is not None:
        # --- НОВЫЙ БЕЗОПАСНЫЙ ПУТЬ (БЕЗ УТЕЧЕК) ---
        # Данные уже содержат чистые 'predicted_signal' из Walk-Forward тест-сетов.
        # Нам не нужно делать запросы к БД и пересчитывать признаки.
        logger.info(f"Используются чистые Out-of-Sample данные ({len(df_valid)} строк) из {os.path.basename(model_path)}")
    else:
        # --- СТАРЫЙ РЕЗЕРВНЫЙ ПУТЬ (для совместимости со старыми моделями и тестами) ---
        logger.warning("Чистые OOS-данные не найдены в файле модели. Переход на резервный in-sample метод...")
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
            df_valid["predicted_signal"] = pd.Series(raw_pred, index=df_valid.index).map(
                signal_map
            )
        else:
            df_valid["predicted_signal"] = raw_pred

    # Набор сеток параметров
    sl_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    tp_grid = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

    results = perform_grid_search(
        df_valid,
        sl_grid,
        tp_grid,
        horizon=settings.LABEL_HORIZON,
        min_trades=settings.CALIBRATION_MIN_TRADES,
    )

    if not results:
        raise ValueError(
            f"Ни одна комбинация SL/TP не набрала {settings.CALIBRATION_MIN_TRADES}+ сделок."
        )

    res_df = pd.DataFrame(results)

    # Сортировка по Sharpe Ratio, а затем по Expectancy
    res_df = res_df.sort_values(
        by=["sharpe_ratio", "expectancy"], ascending=False
    ).reset_index(drop=True)

    best = res_df.iloc[0]

    report = (
        f"⚙️ <b>Результаты автокалибровки рисков:</b>\n"
        f"📉 Stop-Loss (SL): <code>{best['sl_pct']:.1%}</code>\n"
        f"📈 Take-Profit (TP): <code>{best['tp_pct']:.1%}</code>\n"
        f"📊 Коэффициент Шарпа: <code>{best['sharpe_ratio']:.3f}</code>\n"
        f"🎯 Матожидание (Expectancy): <code>{best['expectancy']:.3%}</code> на сделку\n"
        f"💰 Доходность бэктеста: <code>{best['total_return']:.1%}</code> ({best['total_trades']} сделок)"
    )

    return float(best["sl_pct"]), float(best["tp_pct"]), report


async def run_calibration_cli(symbol: str, timeframe: str):
    """Обертка для запуска из консоли."""
    try:
        sl, tp, report = await get_best_calibration(symbol, timeframe)
        # Очистим HTML-теги для красивого вывода в терминал
        clean_report = report.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
        print("\n" + "="*60)
        print(clean_report)
        print("="*60)
        print(f"\nДля сохранения параметров между перезапусками пропишите в .env:\n"
              f"PAPER_SL_PCT={sl}\n"
              f"PAPER_TP_PCT={tp}")
    except Exception as e:
        print(f"[-] Ошибка при калибровке: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    asyncio.run(run_calibration_cli(args.symbol, args.timeframe))