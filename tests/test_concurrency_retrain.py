import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from src.main import _run_retrain_cycle


@pytest.mark.asyncio
async def test_concurrent_retrain_cycles():
    """
    Интеграционный тест: параллельный запуск ретрейна 3 активов в asyncio.gather.
    Проверяет отсутствие взаимных блокировок и корректное распределение потоков.
    """
    mock_settings = MagicMock()
    mock_settings.MIN_KLINES_FOR_TRAIN = 10
    mock_settings.TRAIN_SIZE = 50
    mock_settings.TEST_SIZE = 10
    mock_settings.LABEL_HORIZON = 2
    mock_settings.LABEL_THRESHOLD = 0.01
    mock_settings.ADMIN_IDS = [9999]
    mock_settings.OPTUNA_TUNING_ENABLED = True
    mock_settings.OPTUNA_TRIALS = 2

    mock_settings.get_model_path.side_effect = lambda sym, tf: f"models/saved_models/lgbm_{sym.replace('/', '')}_{tf}.pkl"

    fake_klines = [MagicMock()] * 15

    mock_baseline_result = {"metrics": {"f1": 0.45, "accuracy": 0.5}}
    mock_lgbm_result = {
        "metrics": {"f1": 0.60, "accuracy": 0.65},
        "model_path": "models/staging/lgbm_test.pkl",
    }

    bot_mock = AsyncMock()

    with (
        patch("src.main.get_settings", return_value=mock_settings),
        patch("src.core.db.AsyncSessionFactory") as mock_session_factory,
        patch("src.crud.kline.KlineRepository") as mock_kline_repo_cls,
        patch(
            "src.datasets.build.build_and_save_dataset",
            new_callable=AsyncMock,
            return_value="datasets/test_v2026.parquet",
        ),
        patch(
            "src.models.baseline.run_baseline_experiment",
            new_callable=AsyncMock,
            return_value=mock_baseline_result,
        ),
        patch(
            "src.models.baseline.compute_baseline_holdout_f1",
            new_callable=AsyncMock,
            return_value={
                "f1": mock_baseline_result["metrics"]["f1"],
                "holdout_size": 10,
            },
        ),
        patch(
            "src.models.train.run_lgbm_experiment",
            new_callable=AsyncMock,
            return_value=mock_lgbm_result,
        ),
        patch(
            "scripts.calibrate.get_best_calibration",
            new_callable=AsyncMock,
            return_value=(0.02, 0.04, 5, "cal_report"),
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

        # Конкурентный запуск обучения для трех активов одновременно
        await asyncio.gather(
            _run_retrain_cycle(bot_mock, "BTC/USDT", "1h"),
            _run_retrain_cycle(bot_mock, "ETH/USDT", "1h"),
            _run_retrain_cycle(bot_mock, "SOL/USDT", "1h"),
        )

    # Проверяем, что все 3 асинхронные задачи успешно завершились отправкой отчетов
    assert bot_mock.send_message.call_count == 3