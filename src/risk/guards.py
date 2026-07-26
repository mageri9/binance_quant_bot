from loguru import logger


class RiskGuard:
    """Проверка Circuit Breaker и дневного лимита просадки."""
    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        consecutive_losses_limit: int = 5,
        max_open_positions: int = 3
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.consecutive_losses_limit = consecutive_losses_limit
        self.max_open_positions = max_open_positions

    def validate_order(
        self,
        symbol: str,
        balance_total: float,
        daily_pnl: float,
        consecutive_losses: int,
        open_positions_count: int,
        is_closing: bool = False
    ) -> tuple[bool, str]:
        if is_closing:
            return True, "Ордер закрытия одобрен."

        if consecutive_losses >= self.consecutive_losses_limit:
            reason = f"Circuit Breaker: {consecutive_losses} убытков подряд."
            logger.warning(f"[RiskGuard] {reason}")
            return False, reason

        max_loss = balance_total * self.max_daily_loss_pct
        if daily_pnl < 0 and abs(daily_pnl) >= max_loss:
            reason = f"Превышен суточный лимит потерь (${abs(daily_pnl):.2f} >= ${max_loss:.2f})"
            logger.warning(f"[RiskGuard] {reason}")
            return False, reason

        if open_positions_count >= self.max_open_positions:
            reason = f"Достигнут лимит позиций ({open_positions_count}/{self.max_open_positions})"
            logger.warning(f"[RiskGuard] {reason}")
            return False, reason

        return True, "Проверка рисков пройдена."