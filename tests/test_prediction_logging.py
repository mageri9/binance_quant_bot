import pytest

from src.crud.paper import TradeRepository


@pytest.mark.asyncio
async def test_prediction_log_keeps_economic_diagnostics(temp_db_session):
    prediction = await TradeRepository(temp_db_session).log_prediction(
        symbol="BTC/USDT",
        timeframe="1h",
        candle_time=1_700_000_000_000,
        horizon=5,
        model_id="economic-return-v1",
        price=100.0,
        prediction=0,
        prob_short=0.0,
        prob_hold=0.0,
        prob_long=0.0,
        reason="expected_return_unavailable",
        details={"model_type": "classification"},
    )

    assert prediction.reason == "expected_return_unavailable"
    assert prediction.details == {"model_type": "classification"}


@pytest.mark.asyncio
async def test_prediction_log_remains_compatible_with_probability_protocol(temp_db_session):
    prediction = await TradeRepository(temp_db_session).log_prediction(
        symbol="BTC/USDT",
        model_id="legacy-v1",
        price=100.0,
        prediction=1,
        prob_short=0.1,
        prob_hold=0.2,
        prob_long=0.7,
    )

    assert prediction.reason is None
    assert prediction.details is None
