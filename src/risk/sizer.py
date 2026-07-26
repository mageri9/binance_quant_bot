def calculate_position_size(
    balance: float,
    current_price: float,
    max_allocation_pct: float = 0.10
) -> float:
    """Считает объем ордера в монетах с учетом лимита от банка (по умолчанию 10%)."""
    if balance <= 0 or current_price <= 0:
        return 0.0

    max_capital = balance * max_allocation_pct
    amount = max_capital / current_price
    return round(amount, 6)


def calculate_protection_prices(
    entry_price: float,
    side: str,
    sl_pct: float = 0.02,
    tp_pct: float = 0.04,
    atr_value: float | None = None,
    atr_sl_mult: float = 1.0,
    atr_tp_mult: float = 1.5,
) -> tuple[float, float]:
    """Рассчитывает цены SL/TP (по ATR или процентам)."""
    if atr_value is not None and atr_value > 0:
        sl_dist = atr_value * atr_sl_mult
        tp_dist = atr_value * atr_tp_mult
        if side.lower() in ("buy", "long"):
            return round(entry_price - sl_dist, 2), round(entry_price + tp_dist, 2)
        else:
            return round(entry_price + sl_dist, 2), round(entry_price - tp_dist, 2)

    if side.lower() in ("buy", "long"):
        return round(entry_price * (1 - sl_pct), 2), round(entry_price * (1 + tp_pct), 2)
    else:
        return round(entry_price * (1 + sl_pct), 2), round(entry_price * (1 - tp_pct), 2)