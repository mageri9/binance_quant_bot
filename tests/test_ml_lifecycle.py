from datetime import datetime, timedelta, timezone

import pytest

from src.db.models import PredictionLog, TrainingState
from src.ml.lifecycle import decide_retraining, record_training, resolve_predictions


@pytest.mark.asyncio
async def test_resolved_labels_trigger_retraining(temp_db_session):
    prediction = PredictionLog(
        symbol="BTC/USDT", timeframe="1h", model_id="m1", candle_time=100,
        horizon=2, price=100.0, prediction=1,
        prob_short=0.1, prob_hold=0.2, prob_long=0.7,
    )
    temp_db_session.add(prediction)
    temp_db_session.add(TrainingState(
        symbol="BTC/USDT", timeframe="1h", target="target_triple",
        last_trained_candle=50,
        last_trained_at=datetime.now(timezone.utc) - timedelta(hours=1),
    ))
    await temp_db_session.commit()

    count = await resolve_predictions(
        temp_db_session, symbol="BTC/USDT", timeframe="1h",
        candle_times_and_closes=[(100, 100.0), (200, 101.0), (300, 103.0)],
        threshold=0.01,
    )
    decision = await decide_retraining(
        temp_db_session, symbol="BTC/USDT", timeframe="1h", target="target_triple",
        latest_resolved_candle=300, min_new_labels=1, max_age_hours=168,
    )
    assert count == 1
    assert prediction.true_label == 1
    assert decision.should_train
    assert decision.triggers == ("new_labels",)


@pytest.mark.asyncio
async def test_no_new_candle_never_retrains_on_schedule(temp_db_session):
    temp_db_session.add(TrainingState(
        symbol="BTC/USDT", timeframe="1h", target="target_triple",
        last_trained_candle=300,
        last_trained_at=datetime.now(timezone.utc) - timedelta(days=30),
    ))
    await temp_db_session.commit()
    decision = await decide_retraining(
        temp_db_session, symbol="BTC/USDT", timeframe="1h", target="target_triple",
        latest_resolved_candle=300, min_new_labels=1, max_age_hours=1,
    )
    assert not decision.should_train
    assert "scheduled_control" in decision.triggers


@pytest.mark.asyncio
async def test_rejected_evaluation_advances_retrain_cursor(temp_db_session):
    """A rejected candidate must not retrain the identical labelled window."""
    temp_db_session.add(TrainingState(
        symbol="BTC/USDT", timeframe="1h", target="target_triple",
        last_trained_candle=100,
        last_trained_at=datetime.now(timezone.utc) - timedelta(days=30),
    ))
    await temp_db_session.commit()

    await record_training(
        temp_db_session, symbol="BTC/USDT", timeframe="1h", target="target_triple",
        last_trained_candle=300, dataset_fingerprint="rejected-candidate",
        trigger="new_labels:economic_rejected",
    )

    decision = await decide_retraining(
        temp_db_session, symbol="BTC/USDT", timeframe="1h", target="target_triple",
        latest_resolved_candle=300, min_new_labels=1, max_age_hours=168,
    )

    assert not decision.should_train


@pytest.mark.asyncio
async def test_resolver_uses_candle_time_not_snapshot_arrival_order(temp_db_session):
    prediction = PredictionLog(
        symbol="BTC/USDT", timeframe="1h", model_id="m1", candle_time=100,
        horizon=2, price=100.0, prediction=1,
        prob_short=0.1, prob_hold=0.2, prob_long=0.7,
    )
    temp_db_session.add(prediction)
    await temp_db_session.commit()

    resolved = await resolve_predictions(
        temp_db_session,
        symbol="BTC/USDT",
        timeframe="1h",
        candle_times_and_closes=[(300, 103.0), (100, 100.0), (200, 101.0)],
        threshold=0.01,
    )

    assert resolved == 1
    assert prediction.outcome_price == 103.0
    assert prediction.true_label == 1
