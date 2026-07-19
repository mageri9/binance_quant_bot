import json
from loguru import logger
from redis.asyncio import Redis


class KillSwitchState:
    NORMAL = "NORMAL"
    SAFE_MODE = "SAFE_MODE"
    KILLED = "KILLED"


class KillSwitchManager:
    """
    Управляет аварийным состоянием бота (Kill Switch) в Redis.
    """

    def __init__(self, redis: Redis):
        self.redis = redis

    async def get_state(self) -> tuple[str, str | None, str | None]:
        state = await self.redis.get("nexus:kill_switch:state") or KillSwitchState.NORMAL
        reason = await self.redis.get("nexus:kill_switch:reason")
        details = await self.redis.get("nexus:kill_switch:details")
        return state, reason, details

    async def set_state(self, state: str, reason: str | None = None, details: str | None = None) -> None:
        if state not in [KillSwitchState.NORMAL, KillSwitchState.SAFE_MODE, KillSwitchState.KILLED]:
            raise ValueError(f"Недопустимое состояние Kill Switch: {state}")

        await self.redis.set("nexus:kill_switch:state", state)
        if reason:
            await self.redis.set("nexus:kill_switch:reason", reason)
        else:
            await self.redis.delete("nexus:kill_switch:reason")

        if details:
            await self.redis.set("nexus:kill_switch:details", details)
        else:
            await self.redis.delete("nexus:kill_switch:details")

        logger.warning(f"[KILL SWITCH] Новое состояние: {state} | Причина: {reason or 'нет'} | Детали: {details or 'нет'}")

    async def is_trading_blocked(self) -> bool:
        state, _, _ = await self.get_state()
        return state in [KillSwitchState.SAFE_MODE, KillSwitchState.KILLED]


async def reconcile_positions(
    exchange,
    db_session,
    symbols: list[str],
    kill_switch_manager: KillSwitchManager,
) -> tuple[bool, str | None]:
    """
    Сверяет позиции на бирже (источник истины) с базой данных (кэш).
    При рассинхронизации блокирует бота в SAFE_MODE.
    """
    from src.crud.paper import TradeRepository
    repo = TradeRepository(db_session)

    mismatches = []

    for symbol in symbols:
        ex_pos = await exchange.get_position(symbol)
        db_pos = await repo.get_active_trade(symbol)

        if ex_pos is None:
            if db_pos is not None:
                mismatches.append(
                    f"{symbol}: В БД есть открытая сделка, но на бирже позиция отсутствует."
                )
            continue

        if db_pos is None:
            mismatches.append(
                f"{symbol}: На бирже есть открытая позиция {ex_pos['side']} ({ex_pos['amount']} монет), но в БД сделка отсутствует."
            )
            continue

        ex_side = ex_pos["side"]
        db_side = "SHORT" if db_pos.is_short else "LONG"

        if ex_side != db_side:
            mismatches.append(
                f"{symbol}: Конфликт направления. На бирже: {ex_side}, в БД: {db_side}."
            )
            continue

        if abs(ex_pos["amount"] - db_pos.amount) > 1e-5:
            mismatches.append(
                f"{symbol}: Конфликт объема. На бирже: {ex_pos['amount']:.6f}, в БД: {db_pos.amount:.6f}."
            )
            continue

    if mismatches:
        error_details = "\n".join(mismatches)
        await kill_switch_manager.set_state(
            state=KillSwitchState.SAFE_MODE,
            reason="POSITION_MISMATCH",
            details=error_details,
        )
        return False, error_details

    return True, None