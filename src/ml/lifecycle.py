from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import PredictionLog, TrainingState


@dataclass(frozen=True)
class RetrainDecision:
    should_train: bool
    triggers: tuple[str, ...]
    new_labels: int


async def decide_retraining(
    session: AsyncSession, *, symbol: str, timeframe: str, target: str,
    latest_resolved_candle: int | None, min_new_labels: int,
    max_age_hours: int, drift_detected: bool = False,
    live_metrics_degraded: bool = False,
) -> RetrainDecision:
    state = (await session.execute(select(TrainingState).where(
        TrainingState.symbol == symbol, TrainingState.timeframe == timeframe,
        TrainingState.target == target,
    ))).scalar_one_or_none()
    last_candle = state.last_trained_candle if state else None
    filters = [
        PredictionLog.symbol == symbol,
        PredictionLog.timeframe == timeframe,
        PredictionLog.resolved_at.is_not(None),
    ]
    if last_candle is not None:
        filters.append(PredictionLog.candle_time > last_candle)
    new_labels = int((await session.execute(
        select(func.count(PredictionLog.id)).where(*filters)
    )).scalar_one())
    triggers = []
    if new_labels >= min_new_labels:
        triggers.append("new_labels")
    if drift_detected:
        triggers.append("drift")
    if live_metrics_degraded:
        triggers.append("live_metrics")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    trained_at = state.last_trained_at if state else None
    if trained_at is None or trained_at.replace(tzinfo=trained_at.tzinfo or timezone.utc) <= cutoff:
        triggers.append("scheduled_control")
    # A trigger cannot train unresolved/latest-identical data.
    has_new_observations = latest_resolved_candle is not None and (
        last_candle is None or latest_resolved_candle > last_candle
    )
    return RetrainDecision(bool(triggers) and has_new_observations, tuple(triggers), new_labels)


async def record_training(
    session: AsyncSession, *, symbol: str, timeframe: str, target: str,
    last_trained_candle: int, dataset_fingerprint: str, trigger: str,
) -> TrainingState:
    state = (await session.execute(select(TrainingState).where(
        TrainingState.symbol == symbol, TrainingState.timeframe == timeframe,
        TrainingState.target == target,
    ))).scalar_one_or_none()
    if state is None:
        state = TrainingState(symbol=symbol, timeframe=timeframe, target=target)
        session.add(state)
    state.last_trained_candle = last_trained_candle
    state.last_dataset_fingerprint = dataset_fingerprint
    state.last_trained_at = datetime.now(timezone.utc)
    state.last_trigger = trigger
    await session.commit()
    return state


async def resolve_predictions(
    session: AsyncSession, *, symbol: str, timeframe: str,
    candle_times_and_closes: list[tuple[int, float]], threshold: float = 0.0,
) -> int:
    closes = dict(candle_times_and_closes)
    ordered = [time for time, _ in candle_times_and_closes]
    position = {time: index for index, time in enumerate(ordered)}
    pending = list((await session.execute(select(PredictionLog).where(
        PredictionLog.symbol == symbol, PredictionLog.timeframe == timeframe,
        PredictionLog.resolved_at.is_(None), PredictionLog.candle_time.is_not(None),
    ))).scalars())
    resolved = 0
    for prediction in pending:
        index = position.get(prediction.candle_time)
        if index is None or index + prediction.horizon >= len(ordered):
            continue
        outcome_price = closes[ordered[index + prediction.horizon]]
        realized_return = outcome_price / prediction.price - 1.0
        prediction.outcome_price = outcome_price
        prediction.realized_return = realized_return
        prediction.true_label = 1 if realized_return > threshold else (-1 if realized_return < -threshold else 0)
        prediction.resolved_at = datetime.now(timezone.utc)
        resolved += 1
    if resolved:
        await session.commit()
    return resolved
