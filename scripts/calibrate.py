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
) -> list[dict]:
    """
    Прогоняет сетку параметров SL/TP через симулятор стратегии.
    """
    results = []
    for sl in sl_grid:
        for tp in tp_grid:
            metrics = simulate_strategy(
                df_valid,
                predicted_col="predicted_signal",
                horizon=horizon,
                sl_pct=sl,
                tp_pct=tp,
            )
            if metrics["total_trades"] > 0:
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


async def get_best_calibration(symbol: str, timeframe: str) -> tuple[float, float, str]:
    """
    Проводит калибровку и возвращает: (best_sl, best_tp, formatted_report_text).
    Используется программно внутри retrain_loop.
    """
    settings = get_settings()

    if not os.path.exists(settings.MODEL_PATH):
        raise FileNotFoundError(f"Файл модели {settings.MODEL_PATH} не найден.")

    async with AsyncSessionFactory() as session:
        repo = KlineRepository(session)
        klines = await repo.get_klines(symbol, timeframe, limit=10000)

    if len(klines) < 100:
        raise ValueError(f"Недостаточно исторических данных в БД ({len(klines)} свечей).")

    # Формируем DataFrame
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

    # Расчет признаков
    df_feats = add_features(df)

    with open(settings.MODEL_PATH, "rb") as f:
        saved_data = pickle.load(f)

    model = saved_data["model"]
    features = saved_data["features"]
    scaler = saved_data.get("scaler")

    df_valid = df_feats.dropna(subset=features).copy()
    if df_valid.empty:
        raise ValueError("После расчета признаков не осталось валидных данных.")

    X = df_valid[features]
    if scaler is not None:
        X = scaler.transform(X)

    # Записываем исторические сигналы
    df_valid["predicted_signal"] = model.predict(X)

    # Набор сеток параметров
    sl_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    tp_grid = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

    results = perform_grid_search(df_valid, sl_grid, tp_grid, horizon=settings.LABEL_HORIZON)

    if not results:
        raise ValueError("Не совершено ни одной сделки при симуляции.")

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