import pandas as pd
import numpy as np

from src.execution.kernel import ExecutionCosts, ExecutionKernel


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
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "total_return": 0.0,
        }

    # Database financial fields use Decimal, while NumPy metrics are float-domain.
    trades_arr = np.asarray(trades, dtype=float)

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
    # Approximate 95% CI for non-annualized Sharpe; conservative lower bound is
    # used during tuning so a few lucky trades cannot dominate selection.
    if total_trades > 1:
        sharpe_se = float(np.sqrt((1.0 + 0.5 * sharpe_ratio**2) / total_trades))
        sharpe_ci_low = sharpe_ratio - 1.96 * sharpe_se
        sharpe_ci_high = sharpe_ratio + 1.96 * sharpe_se
    else:
        sharpe_ci_low = sharpe_ci_high = 0.0

    # 4. Коэффициент Сортино (риск только на убыточных сделках)
    downside_returns = trades_arr[trades_arr < 0]
    downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0.0

    # 5. Математическое ожидание одной сделки
    expectancy = mean_return

    # 6. Расчет кривой капитала (Equity) и Максимальной Просадки (Max Drawdown)
    # Начинаем с баланса 1.0 (100%)
    equity_curve = [1.0]
    for r in trades_arr:
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
        "sharpe_ci_low": sharpe_ci_low,
        "sharpe_ci_high": sharpe_ci_high,
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
    return_trade_log: bool = False,
    execution_kernel: ExecutionKernel | None = None,
) -> dict | tuple[dict, pd.DataFrame]:
    """
    ...(докстринг как был)...

    sl_atr_mult / tp_atr_mult: если оба заданы и в df есть колонка 'atr',
    барьеры SL/TP считаются как entry_price -/+ mult * ATR(на момент входа)
    вместо фиксированных sl_pct/tp_pct. Если ATR в момент входа NaN или <= 0,
    сделка использует фолбэк на sl_pct/tp_pct для этой конкретной сделки.
    """
    prices = df["close"].values
    # A closed-candle signal can only be acted on at the next candle's open.
    # Old artifacts without OHLC opens remain readable, but production data has it.
    opens = df["open"].values if "open" in df.columns else prices
    highs = df["high"].values
    lows = df["low"].values
    signals = df[predicted_col].values
    atr_values = df["atr"].values if "atr" in df.columns else None
    use_atr_barrier = (
        sl_atr_mult is not None and tp_atr_mult is not None and atr_values is not None
    )
    n = len(df)

    # Preserve transaction_cost for callers while sharing fill math with paper.
    kernel = execution_kernel or ExecutionKernel(
        ExecutionCosts(
            commission_rate=0.0,
            slippage_rate=transaction_cost,
            bid_ask_spread_rate=0.0,
            funding_rate_per_trade=0.0,
        )
    )
    trades = []
    trade_log = []

    position_type = None
    entry_price = 0.0
    entry_idx = 0
    sl_price = 0.0
    tp_price = 0.0
    exit_at_next_open = False

    def _close_trade(exit_idx, exit_reference_price):
        entry_fill = kernel.market_fill(
            side="sell" if position_type == "SHORT" else "buy",
            reference_price=entry_price,
            amount=1,
        )
        exit_fill = kernel.market_fill(
            side="buy" if position_type == "SHORT" else "sell",
            reference_price=exit_reference_price,
            amount=1,
        )
        trade_return = float(kernel.realized_return(
            entry=entry_fill, exit=exit_fill, is_short=position_type == "SHORT",
        ))
        trades.append(trade_return)
        if return_trade_log:
            trade_log.append(
                {
                    "entry_idx": entry_idx,
                    "exit_idx": exit_idx,
                    "side": position_type,
                    "return": trade_return,
                }
            )

    for i in range(n):
        if position_type is not None and exit_at_next_open:
            _close_trade(i, opens[i])
            position_type = None
            exit_at_next_open = False
            continue

        if position_type is None:
            if i > 0:
                signal_idx = i - 1
                atr_at_entry = atr_values[signal_idx] if use_atr_barrier else None
                atr_ok = (
                    atr_at_entry is not None
                    and not np.isnan(atr_at_entry)
                    and atr_at_entry > 0
                )

                if signals[signal_idx] == 1:
                    position_type = "LONG"
                    entry_price = opens[i]
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

                elif signals[signal_idx] == -1:
                    position_type = "SHORT"
                    entry_price = opens[i]
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
        if position_type == "LONG":
            if lows[i] <= sl_price:
                _close_trade(i, sl_price)
                position_type = None
                continue
            if highs[i] >= tp_price:
                _close_trade(i, tp_price)
                position_type = None
                continue
            if i - entry_idx >= horizon:
                exit_at_next_open = True
                continue

        elif position_type == "SHORT":
            if highs[i] >= sl_price:
                _close_trade(i, sl_price)
                position_type = None
                continue
            if lows[i] <= tp_price:
                _close_trade(i, tp_price)
                position_type = None
                continue
            if i - entry_idx >= horizon:
                exit_at_next_open = True
                continue

    metrics = calculate_strategy_metrics(trades)
    if return_trade_log:
        trades_df = pd.DataFrame(trade_log, columns=["entry_idx", "exit_idx", "side", "return"])
        return metrics, trades_df
    return metrics
