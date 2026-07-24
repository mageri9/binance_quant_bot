import pickle
import os

import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.main import _run_retrain_cycle


@pytest.mark.asyncio
async def test_retrain_cycle_uses_economic_metrics_not_baseline_f1(temp_db_session, tmp_path):
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
    staging_model_path = str(tmp_path / "staging" / "lgbm_BTCUSDT_1h.pkl")
    os.makedirs(os.path.dirname(staging_model_path), exist_ok=True)
    with open(staging_model_path, "wb") as f:
        pickle.dump({"model": "dummy"}, f)
    mock_lgbm_result = {
        "metrics": {"f1": new_f1_value, "accuracy": 0.65},
        "model_path": staging_model_path,
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
            "src.models.baseline.compute_baseline_holdout_f1",
            new_callable=AsyncMock,
            return_value={"f1": baseline_f1_value, "holdout_size": 20},
        ) as mock_compute_holdout,
        patch(
            "src.models.train.run_lgbm_experiment",
            new_callable=AsyncMock,
            return_value=mock_lgbm_result,
        ) as mock_run_lgbm,
        patch(
            "scripts.calibrate.get_best_calibration",
            new_callable=AsyncMock,
            return_value=(
                0.02,
                0.04,
                5,
                "report",
                {
                    "sharpe_ratio": 0.5,
                    "expectancy": 0.01,
                    "total_return": 0.05,
                    "total_trades": 20,
                    "win_rate": 0.55,
                    "profit_factor": 1.3,
                    "sortino_ratio": 0.4,
                    "max_drawdown": 0.08,
                },
            ),
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

        # run_lgbm_experiment должен быть вызван с честным (apples-to-apples)
        # baseline_f1 из compute_baseline_holdout_f1, а не с неопределённым значением
        assert mock_run_lgbm.called
        _, called_kwargs = mock_run_lgbm.call_args
        assert "baseline_f1" not in called_kwargs
        assert not mock_compute_holdout.called

    # Уведомление админу должно уйти (новая модель лучше baseline)
    bot_mock.send_message.assert_called_once()
    sent_text = bot_mock.send_message.call_args[1]["text"]
    assert "ПРИНЯТА В ПРОД" in sent_text

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
        patch(
            "src.datasets.build.build_and_save_dataset",
            new_callable=AsyncMock,
            return_value="ds.parquet",
        ),
        patch(
            "src.models.baseline.run_baseline_experiment",
            new_callable=AsyncMock,
            return_value=mock_baseline_result,
        ),
        patch(
            "src.models.baseline.compute_baseline_holdout_f1",
            new_callable=AsyncMock,
            return_value={"f1": 0.30, "holdout_size": 20},
        ),
        patch(
            "src.models.train.run_lgbm_experiment",
            new_callable=AsyncMock,
            return_value=mock_lgbm_result,
        ),
        patch(
            "scripts.calibrate.get_best_calibration",
            new_callable=AsyncMock,
            return_value=(
                0.02,
                0.04,
                5,
                "report",
                {
                    "sharpe_ratio": 0.5,
                    "expectancy": 0.01,
                    "total_return": 0.05,
                    "total_trades": 20,
                    "win_rate": 0.55,
                    "profit_factor": 1.3,
                    "sortino_ratio": 0.4,
                    "max_drawdown": 0.08,
                },
            ),
        ),
        patch(
            "os.path.exists",
            side_effect=lambda p: p == staging_oos_path or p == "ds.parquet",
        ),
        patch("pandas.read_parquet", return_value=pd.DataFrame({"close": [1.0]})),
    ):
        mock_kline_repo_cls.return_value.get_klines = AsyncMock(return_value=fake_klines)
        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = MagicMock()
        mock_session_factory.return_value = session_ctx

        await _run_retrain_cycle(bot_mock, "BTC/USDT", "1h")

    prod_oos_path = get_oos_path(prod_model_path)
    assert os.path.exists(prod_oos_path)

@pytest.mark.asyncio
async def test_retrain_cycle_blocked_by_economic_gate(temp_db_session, tmp_path):
    """
    Регрессионный тест на Economic Quality Gate: модель может пройти F1-гейт
    (holdout_f1 > baseline), но обязана быть отклонена, если её честные
    (held-out) метрики прибыльности хуже текущей прод-модели.
    """
    import pickle

    prod_model_path = str(tmp_path / "lgbm_BTCUSDT_1h.pkl")

    # Существующая прод-модель с сохранёнными backtest_metrics (уже прошедшая
    # через квест 7/8 на прошлом цикле ретрейна) — хороший Sharpe.
    with open(prod_model_path, "wb") as f:
        pickle.dump({
            "model": "dummy_prod_model",
            "backtest_metrics": {
                "sharpe_ratio": 0.8,
                "expectancy": 0.015,
                "total_return": 0.10,
                "total_trades": 30,
                "win_rate": 0.55,
                "profit_factor": 1.4,
                "sortino_ratio": 0.7,
                "max_drawdown": 0.06,
            },
        }, f)

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
        pickle.dump({"model": "dummy_candidate"}, f)

    mock_baseline_result = {"metrics": {"f1": 0.30, "accuracy": 0.5}}
    mock_lgbm_result = {
        "metrics": {"f1": 0.55, "holdout_f1": 0.55, "accuracy": 0.6},
        "model_path": staging_model_path,
    }

    # Кандидат проходит F1-гейт, но его честный Sharpe хуже прод-модели (0.8)
    candidate_honest_metrics = {
        "sharpe_ratio": 0.2,
        "expectancy": 0.005,
        "total_return": 0.02,
        "total_trades": 20,
        "win_rate": 0.45,
        "profit_factor": 1.05,
        "sortino_ratio": 0.15,
        "max_drawdown": 0.12,
    }

    bot_mock = AsyncMock()

    with (
        patch("src.main.get_settings", return_value=mock_settings),
        patch("src.core.db.AsyncSessionFactory") as mock_session_factory,
        patch("src.crud.kline.KlineRepository") as mock_kline_repo_cls,
        patch(
            "src.datasets.build.build_and_save_dataset",
            new_callable=AsyncMock,
            return_value="ds.parquet",
        ),
        patch(
            "src.models.baseline.run_baseline_experiment",
            new_callable=AsyncMock,
            return_value=mock_baseline_result,
        ),
        patch(
            "src.models.baseline.compute_baseline_holdout_f1",
            new_callable=AsyncMock,
            return_value={"f1": 0.30, "holdout_size": 20},
        ),
        patch(
            "src.models.train.run_lgbm_experiment",
            new_callable=AsyncMock,
            return_value=mock_lgbm_result,
        ),
        patch(
            "scripts.calibrate.get_best_calibration",
            new_callable=AsyncMock,
            return_value=(0.02, 0.04, 5, "report", candidate_honest_metrics),
        ),
        patch("pandas.read_parquet"),
    ):
        mock_kline_repo_cls.return_value.get_klines = AsyncMock(return_value=fake_klines)
        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = MagicMock()
        mock_session_factory.return_value = session_ctx

        await _run_retrain_cycle(bot_mock, "BTC/USDT", "1h")

    # Прод-модель НЕ должна была быть перезаписана кандидатом
    with open(prod_model_path, "rb") as f:
        artifact_after = pickle.load(f)
    assert artifact_after["model"] == "dummy_prod_model"

    # Админу должно уйти сообщение об отклонении Economic Gate
    bot_mock.send_message.assert_called_once()
    sent_text = bot_mock.send_message.call_args[1]["text"]
    assert "Economic Gate" in sent_text
    assert "ОТКЛОНЕНА" in sent_text
