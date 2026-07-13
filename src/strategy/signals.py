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
) -> dict:
    """
    Симулирует торговлю на истории (Long-only).

    Вход в позицию: когда predicted_signal == 1 и мы свободны.
    Выход: при срабатывании Stop-Loss, Take-Profit или принудительно по времени (horizon).
    Учитывает комиссию биржи на вход и выход.
    """
    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    signals = df[predicted_col].values
    n = len(df)

    trades = []  # Хранит доходность по каждой совершенной сделке

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    sl_price = 0.0
    tp_price = 0.0

    for i in range(n):
        if not in_position:
            # Ищем сигнал на вход
            if signals[i] == 1 and (i < n - 1):
                in_position = True
                entry_price = prices[i]
                entry_idx = i

                # Задаем уровни SL / TP относительно цены входа
                sl_price = entry_price * (1.0 - sl_pct) if sl_pct is not None else 0.0
                tp_price = (
                    entry_price * (1.0 + tp_pct) if tp_pct is not None else float("inf")
                )
        else:
            # Мы находимся в сделке, проверяем условия выхода:

            # 1. Проверяем срабатывание Stop-Loss по минимальной цене свечи
            if lows[i] <= sl_price:
                # Фиксируем убыток по цене SL с учетом комиссий биржи за покупку и продажу
                trade_return = (sl_price - entry_price) / entry_price - (
                    2 * transaction_cost
                )
                trades.append(trade_return)
                in_position = False
                continue

            # 2. Проверяем срабатывание Take-Profit по максимальной цене свечи
            if highs[i] >= tp_price:
                # Фиксируем прибыль по цене TP с учетом комиссий биржи
                trade_return = (tp_price - entry_price) / entry_price - (
                    2 * transaction_cost
                )
                trades.append(trade_return)
                in_position = False
                continue

            # 3. Выход по истечении горизонта времени (Time Exit)
            if i - entry_idx >= horizon:
                # Выходим по цене закрытия текущей свечи
                trade_return = (prices[i] - entry_price) / entry_price - (
                    2 * transaction_cost
                )
                trades.append(trade_return)
                in_position = False
                continue

    return calculate_strategy_metrics(trades)