import pandas as pd
import numpy as np


def calculate_strategy_metrics(trades: list[float]) -> dict:
    """
    Рассчитывает профессиональные трейдерские показатели по списку доходностей сделок.
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "total_return": 0.0,
        }

    trades_arr = np.array(trades)

    total_trades = len(trades_arr)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr <= 0]

    # 1. Доля прибыльных сделок (Win Rate)
    win_rate = float(len(wins) / total_trades)

    # 2. Профит Фактор (Валовая прибыль / Валовый убыток)
    gross_profits = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = (
        gross_profits / gross_losses
        if gross_losses > 0
        else (gross_profits if gross_profits > 0 else 1.0)
    )

    # 3. Коэффициент Шарпа по сделкам (средний доход на сделку к риску)
    mean_return = float(np.mean(trades_arr))
    std_return = float(np.std(trades_arr))
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

    # 4. Коэффициент Сортино (риск только на убыточных сделках)
    downside_returns = trades_arr[trades_arr < 0]
    downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0

    # 5. Математическое ожидание одной сделки
    expectancy = mean_return

    # 6. Расчет кривой капитала (Equity) и Максимальной Просадки (Max Drawdown)
    # Начинаем с баланса 1.0 (100%)
    equity_curve = [1.0]
    for r in trades:
        equity_curve.append(equity_curve[-1] * (1.0 + r))

    equity_curve = np.array(equity_curve)

    # Рассчитываем пики кривой баланса для поиска просадок
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    max_drawdown = float(np.max(drawdowns))

    total_return = float(equity_curve[-1] - 1.0)

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "expectancy": expectancy,
        "total_return": total_return,
    }


def simulate_strategy(
    df: pd.DataFrame,
    predicted_col: str = "predicted_signal",
    horizon: int = 5,
    sl_pct: float | None = 0.02,
    tp_pct: float | None = 0.04,
    transaction_cost: float = 0.001,
    sl_atr_mult: float | None = None,
    tp_atr_mult: float | None = None,
) -> dict:
    """
    ...(докстринг как был)...

    sl_atr_mult / tp_atr_mult: если оба заданы и в df есть колонка 'atr',
    барьеры SL/TP считаются как entry_price -/+ mult * ATR(на момент входа)
    вместо фиксированных sl_pct/tp_pct. Если ATR в момент входа NaN или <= 0,
    сделка использует фолбэк на sl_pct/tp_pct для этой конкретной сделки.
    """
    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    signals = df[predicted_col].values
    atr_values = df["atr"].values if "atr" in df.columns else None
    use_atr_barrier = (
        sl_atr_mult is not None and tp_atr_mult is not None and atr_values is not None
    )
    n = len(df)

    trades = []

    position_type = None
    entry_price = 0.0
    entry_idx = 0
    sl_price = 0.0
    tp_price = 0.0

    for i in range(n):
        if position_type is None:
            if i < n - 1:
                atr_at_entry = atr_values[i] if use_atr_barrier else None
                atr_ok = (
                    atr_at_entry is not None
                    and not np.isnan(atr_at_entry)
                    and atr_at_entry > 0
                )

                if signals[i] == 1:
                    position_type = "LONG"
                    entry_price = prices[i]
                    entry_idx = i
                    if atr_ok:
                        sl_price = entry_price - sl_atr_mult * atr_at_entry
                        tp_price = entry_price + tp_atr_mult * atr_at_entry
                    else:
                        sl_price = (
                            entry_price * (1.0 - sl_pct)
                            if sl_pct is not None
                            else float("-inf")
                        )
                        tp_price = (
                            entry_price * (1.0 + tp_pct)
                            if tp_pct is not None
                            else float("inf")
                        )

                elif signals[i] == -1:
                    position_type = "SHORT"
                    entry_price = prices[i]
                    entry_idx = i
                    if atr_ok:
                        sl_price = entry_price + sl_atr_mult * atr_at_entry
                        tp_price = entry_price - tp_atr_mult * atr_at_entry
                    else:
                        sl_price = (
                            entry_price * (1.0 + sl_pct)
                            if sl_pct is not None
                            else float("inf")
                        )
                        tp_price = (
                            entry_price * (1.0 - tp_pct)
                            if tp_pct is not None
                            else float("-inf")
                        )
        else:
            # --- остальная логика удержания/выхода БЕЗ ИЗМЕНЕНИЙ ---

            if position_type == "LONG":
                # 1. Проверяем Stop-Loss по минимальной цене свечи
                if lows[i] <= sl_price:
                    trade_return = (sl_price - entry_price) / entry_price - (
                        2 * transaction_cost
                    )
                    trades.append(trade_return)
                    position_type = None
                    continue

                # 2. Проверяем Take-Profit по максимальной цене свечи
                if highs[i] >= tp_price:
                    trade_return = (tp_price - entry_price) / entry_price - (
                        2 * transaction_cost
                    )
                    trades.append(trade_return)
                    position_type = None
                    continue

                # 3. Выход по истечении горизонта времени (Time Exit)
                if i - entry_idx >= horizon:
                    trade_return = (prices[i] - entry_price) / entry_price - (
                        2 * transaction_cost
                    )
                    trades.append(trade_return)
                    position_type = None
                    continue

            elif position_type == "SHORT":
                # 1. Проверяем Stop-Loss (для SHORT это рост цены вверх)
                if highs[i] >= sl_price:
                    trade_return = (entry_price - sl_price) / entry_price - (
                        2 * transaction_cost
                    )
                    trades.append(trade_return)
                    position_type = None
                    continue

                # 2. Проверяем Take-Profit (для SHORT это падение цены вниз)
                if lows[i] <= tp_price:
                    trade_return = (entry_price - tp_price) / entry_price - (
                        2 * transaction_cost
                    )
                    trades.append(trade_return)
                    position_type = None
                    continue

                # 3. Выход по истечении горизонта времени (Time Exit)
                if i - entry_idx >= horizon:
                    trade_return = (entry_price - prices[i]) / entry_price - (
                        2 * transaction_cost
                    )
                    trades.append(trade_return)
                    position_type = None
                    continue

    return calculate_strategy_metrics(trades)