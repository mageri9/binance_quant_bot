import pickle
import os

import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.main import _run_retrain_cycle


@pytest.mark.asyncio
async def test_retrain_cycle_computes_baseline_before_lgbm_call(temp_db_session):
    """
    Регрессионный тест на баг: baseline_f1 использовался в вызове
    run_lgbm_experiment раньше своего присваивания -> UnboundLocalError
    на каждой попытке автопереобучения.

    Проверяем, что:
    1) _run_retrain_cycle не падает с UnboundLocalError;
    2) run_lgbm_experiment вызывается с baseline_f1, реально взятым
       из результата run_baseline_experiment (а не None/не мусор).
    """
    mock_settings = MagicMock()
    mock_settings.MIN_KLINES_FOR_TRAIN = 5
    mock_settings.TRAIN_SIZE = 100
    mock_settings.TEST_SIZE = 20
    mock_settings.LABEL_HORIZON = 5
    mock_settings.LABEL_THRESHOLD = 0.01
    mock_settings.ADMIN_IDS = [111]
    mock_settings.get_model_path.return_value = "models/saved_models/lgbm_BTCUSDT_1h.pkl"

    fake_klines = [MagicMock()] * 10  # длина >= MIN_KLINES_FOR_TRAIN

    baseline_f1_value = 0.42
    new_f1_value = 0.55  # новая модель лучше baseline -> ветка продвижения в прод

    mock_baseline_result = {"metrics": {"f1": baseline_f1_value, "accuracy": 0.6}}
    mock_lgbm_result = {
        "metrics": {"f1": new_f1_value, "accuracy": 0.65},
        "model_path": "models/staging/lgbm_BTCUSDT_1h.pkl",
    }

    bot_mock = AsyncMock()

    with (
        patch("src.main.get_settings", return_value=mock_settings),
        patch("src.core.db.AsyncSessionFactory") as mock_session_factory,
        patch("src.crud.kline.KlineRepository") as mock_kline_repo_cls,
        patch(
            "src.datasets.build.build_and_save_dataset",
            new_callable=AsyncMock,
            return_value="datasets/BTCUSDT_1h_v20260101.parquet",
        ),
        patch(
            "src.models.baseline.run_baseline_experiment",
            new_callable=AsyncMock,
            return_value=mock_baseline_result,
        ) as mock_run_baseline,
        patch(
            "src.models.train.run_lgbm_experiment",
            new_callable=AsyncMock,
            return_value=mock_lgbm_result,
        ) as mock_run_lgbm,
        patch(
            "scripts.calibrate.get_best_calibration",
            new_callable=AsyncMock,
            return_value=(0.02, 0.04, 5, "report"),
        ),
        patch("os.path.exists", return_value=False),
        patch("shutil.copy"),
        patch("os.replace"),
        patch("pandas.read_parquet"),
    ):
        mock_kline_repo_cls.return_value.get_klines = AsyncMock(return_value=fake_klines)

        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = MagicMock()
        mock_session_factory.return_value = session_ctx

        # Не должно упасть с UnboundLocalError
        await _run_retrain_cycle(bot_mock, "BTC/USDT", "1h")

    # run_lgbm_experiment должен быть вызван с корректным baseline_f1,
    # а не с неопределённым значением
    assert mock_run_lgbm.called
    _, called_kwargs = mock_run_lgbm.call_args
    assert called_kwargs["baseline_f1"] == pytest.approx(baseline_f1_value)

    # Уведомление админу должно уйти (новая модель лучше baseline)
    bot_mock.send_message.assert_called_once()
    sent_text = bot_mock.send_message.call_args[1]["text"]
    assert "обновлена в продакшне" in sent_text

@pytest.mark.asyncio
async def test_retrain_cycle_copies_oos_parquet_to_production(temp_db_session, tmp_path):
    """
    При успешном промоушене staging-модели в прод sibling OOS-parquet
    должен копироваться вместе с pickle-артефактом модели.
    """
    from src.utils.artifact_paths import get_oos_path

    prod_model_path = str(tmp_path / "lgbm_BTCUSDT_1h.pkl")

    mock_settings = MagicMock()
    mock_settings.MIN_KLINES_FOR_TRAIN = 5
    mock_settings.TRAIN_SIZE = 100
    mock_settings.TEST_SIZE = 20
    mock_settings.LABEL_HORIZON = 5
    mock_settings.LABEL_THRESHOLD = 0.01
    mock_settings.ADMIN_IDS = [111]
    mock_settings.get_model_path.return_value = prod_model_path

    fake_klines = [MagicMock()] * 10

    staging_model_path = str(tmp_path / "staging" / "lgbm_BTCUSDT_1h.pkl")
    os.makedirs(os.path.dirname(staging_model_path), exist_ok=True)
    with open(staging_model_path, "wb") as f:
        pickle.dump({"model": "dummy"}, f)
    staging_oos_path = get_oos_path(staging_model_path)
    pd.DataFrame({"close": [1.0, 2.0]}).to_parquet(staging_oos_path, index=False)

    mock_baseline_result = {"metrics": {"f1": 0.30, "accuracy": 0.5}}
    mock_lgbm_result = {
        "metrics": {"f1": 0.55, "accuracy": 0.6},
        "model_path": staging_model_path,
    }

    bot_mock = AsyncMock()

    with (
        patch("src.main.get_settings", return_value=mock_settings),
        patch("src.core.db.AsyncSessionFactory") as mock_session_factory,
        patch("src.crud.kline.KlineRepository") as mock_kline_repo_cls,
        patch("src.datasets.build.build_and_save_dataset", new_callable=AsyncMock, return_value="ds.parquet"),
        patch("src.models.baseline.run_baseline_experiment", new_callable=AsyncMock, return_value=mock_baseline_result),
        patch("src.models.train.run_lgbm_experiment", new_callable=AsyncMock, return_value=mock_lgbm_result),
        patch("scripts.calibrate.get_best_calibration", new_callable=AsyncMock, return_value=(0.02, 0.04, 5, "report")),
        patch("os.path.exists", side_effect=lambda p: p == staging_oos_path or p == "ds.parquet"),
        patch("pandas.read_parquet", return_value=pd.DataFrame({"close": [1.0]})),
    ):
        mock_kline_repo_cls.return_value.get_klines = AsyncMock(return_value=fake_klines)
        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = MagicMock()
        mock_session_factory.return_value = session_ctx

        await _run_retrain_cycle(bot_mock, "BTC/USDT", "1h")

    prod_oos_path = get_oos_path(prod_model_path)
    assert os.path.exists(prod_oos_path)