from enum import Enum


class RiskDecision(Enum):
    OPEN = "OPEN"
    DENY = "DENY"
    REDUCE_SIZE = "REDUCE_SIZE"
    FORCE_CLOSE = "FORCE_CLOSE"


class RiskEngine:
    """
    Защитный модуль управления рисками (Risk Engine).
    """

    def __init__(
        self,
        max_allocation_pct: float = 0.15,
        max_open_positions: int = 3,
        max_daily_loss_pct: float = 0.05,
        consecutive_losses_limit: int = 5,
    ):
        self.max_allocation_pct = max_allocation_pct
        self.max_open_positions = max_open_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.consecutive_losses_limit = consecutive_losses_limit

    async def validate_signal(
        self,
        symbol: str,
        side: str,
        requested_amount: float,
        current_price: float,
        balance_free: float,
        balance_total: float,
        open_positions: list[dict],
        closed_trades_last_24h: list[dict],
        consecutive_losses: int,
    ) -> tuple[RiskDecision, float, str]:
        # 1. Защита Circuit Breaker
        if consecutive_losses >= self.consecutive_losses_limit:
            return (
                RiskDecision.DENY,
                0.0,
                f"Circuit Breaker активирован: превышена серия из {self.consecutive_losses_limit} убытков.",
            )

        # 2. Суточный лимит просадки
        total_pnl_24h = sum(t.get("pnl", 0.0) for t in closed_trades_last_24h if t.get("pnl") is not None)
        max_allowed_loss = balance_total * self.max_daily_loss_pct
        if total_pnl_24h < 0 and abs(total_pnl_24h) >= max_allowed_loss:
            return (
                RiskDecision.DENY,
                0.0,
                f"Превышен лимит потерь за 24ч: PnL {total_pnl_24h:.2f}$ (лимит: {max_allowed_loss:.2f}$)",
            )

        # 3. Лимит открытых позиций
        is_already_open = any(pos["symbol"] == symbol for pos in open_positions)
        if not is_already_open and len(open_positions) >= self.max_open_positions:
            return (
                RiskDecision.DENY,
                0.0,
                f"Достигнут лимит открытых позиций: {len(open_positions)}/{self.max_open_positions}",
            )

        # Безусловное одобрение закрытия существующей сделки
        if is_already_open:
            active_pos = next(pos for pos in open_positions if pos["symbol"] == symbol)
            is_close_order = (side.lower() == "sell" and active_pos["side"] == "LONG") or \
                             (side.lower() == "buy" and active_pos["side"] == "SHORT")
            if is_close_order:
                return RiskDecision.OPEN, active_pos["amount"], "Ордер закрытия позиции одобрен."

        # 4. Проверка максимального размера сделки
        requested_value = requested_amount * current_price
        max_allowed_value = balance_total * self.max_allocation_pct

        adjusted_amount = requested_amount
        decision = RiskDecision.OPEN
        reason = "Сделка полностью одобрена."

        if requested_value > max_allowed_value:
            adjusted_amount = max_allowed_value / current_price
            decision = RiskDecision.REDUCE_SIZE
            reason = f"Объем ордера снижен с {requested_value:.2f}$ до лимита {max_allowed_value:.2f}$"

        # 5. Проверка фактической нехватки свободного кэша
        final_value = adjusted_amount * current_price
        if balance_free < final_value:
            adjusted_amount = balance_free / current_price
            decision = RiskDecision.REDUCE_SIZE
            reason = f"Объем ордера снижен до доступного кэша: {balance_free:.2f}$"

        # Фильтр микро-сделок
        if adjusted_amount * current_price < 1.0:
            return RiskDecision.DENY, 0.0, "Объем ордера меньше минимально допустимого ($1.0)."

        return decision, adjusted_amount, reason