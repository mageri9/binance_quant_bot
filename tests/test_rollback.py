import os
import shutil
import tempfile
import pytest
import pickle
from unittest.mock import AsyncMock, patch, MagicMock
from aiogram import Bot

from src.main import check_and_rollback_model, _get_backup_timestamp, _rotate_backups
from src.crud.paper import PaperTradingRepository


@pytest.mark.asyncio
async def test_check_and_rollback_model_degradation(temp_db_session):
    # Создаем временную директорию для симуляции папки моделей
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "lgbm_BTCUSDT_1h.pkl")
        backup_path = os.path.join(tmpdir, "lgbm_BTCUSDT_1h_backup_202607140000.pkl")

        # Создаем валидный тестовый упакованный артефакт
        test_artifact = {
            "model_id": "lgbm_BTCUSDT_1h_backup_test_v1",
            "model": "dummy_model_object_for_test",
            "calibration": {
                "sl_pct": 0.015,
                "tp_pct": 0.035
            }
        }

        # Записываем его как в прод-файл, так и в файл бэкапа
        with open(model_path, "wb") as f:
            pickle.dump(test_artifact, f)
        with open(backup_path, "wb") as f:
            pickle.dump(test_artifact, f)

        # Настраиваем фиктивные параметры конфигурации
        mock_settings = MagicMock()
        mock_settings.MODEL_PATH = model_path
        mock_settings.ADMIN_IDS = [12345]
        mock_settings.ROLLBACK_CHECK_WINDOW = 3
        mock_settings.ROLLBACK_WIN_RATE_THRESHOLD = 0.40
        mock_settings.ROLLBACK_MAX_DRAWDOWN_THRESHOLD = 0.10

        # Обучаем мок возвращать правильный путь для временной модели (Quest 9)
        mock_settings.get_model_path.return_value = model_path

        # Имитируем в БД 3 убыточные сделки (win_rate = 0%, drawdown большой)
        repo = PaperTradingRepository(temp_db_session)
        for i in range(3):
            trade = await repo.create_trade(
                symbol="BTC/USDT",
                entry_price=100.0,
                amount=1.0,
                sl_price=90.0,
                tp_price=110.0,
                entry_candle_time=1000 + i,
                is_short=False
            )
            # Фиксируем убыток по SL (90.0)
            await repo.close_trade(trade, exit_price=90.0, pnl=-10.0)

        bot_mock = AsyncMock(spec=Bot)

        # Патчим импорт get_settings и get_redis в src.main
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch("src.main.get_redis") as mock_redis_func,
        ):
            # Настраиваем заглушку для Redis
            redis_mock = AsyncMock()
            redis_mock.get.return_value = None  # Кулдаун пуст
            mock_redis_func.return_value = redis_mock

            # Запускаем проверку SRE
            await check_and_rollback_model(temp_db_session, bot_mock, "BTC/USDT", "1h")

            # Проверки:
            # 1. Должно отправиться критическое оповещение администраторам
            bot_mock.send_message.assert_called_once()
            alert_text = bot_mock.send_message.call_args[1]["text"]

            # Проверяем, что алерт содержит расширенные метаданные
            assert "деградировали" in alert_text.lower()
            assert "откат успешно выполнен" in alert_text.lower()
            assert "v1" in alert_text.lower()  # Наличие ID модели в логе
            assert "1.5%" in alert_text.lower()  # Наличие восстановленного SL

            # 2. Файл текущей модели должен быть успешно переписан стабильной копией
            with open(model_path, "rb") as f:
                content = pickle.load(f)
            assert content["model_id"] == "lgbm_BTCUSDT_1h_backup_test_v1"


def test_get_backup_timestamp_parsing():
    """Проверяет извлечение временной метки из различных форматов путей файлов бэкапов."""
    filepath = "/some/dir/lgbm_BTCUSDT_1h_backup_202607151030.pkl"
    ts = _get_backup_timestamp(filepath)
    assert ts == 202607151030

    # Ошибочный или поврежденный путь
    invalid_filepath = "/some/dir/lgbm_BTCUSDT_1h_backup_abc.pkl"
    assert _get_backup_timestamp(invalid_filepath) == 0


def test_backup_rotation():
    """Проверяет правильность ротации файлов бэкапов на диске."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем 7 пустых бэкапов с разными временными метками
        timestamps = [
            "202607151000",
            "202607151005",
            "202607151010",
            "202607151015",
            "202607151020",
            "202607151025",
            "202607151030",
        ]
        symbol = "BTCUSDT"
        tf = "1h"

        for ts in timestamps:
            filename = f"lgbm_{symbol}_{tf}_backup_{ts}.pkl"
            with open(os.path.join(tmpdir, filename), "w") as f:
                f.write("test")

        # Оставляем последние 5
        _rotate_backups(tmpdir, symbol, tf, keep_count=5)

        remaining_files = os.listdir(tmpdir)
        assert len(remaining_files) == 5

        # Старые файлы должны быть удалены
        assert f"lgbm_{symbol}_{tf}_backup_202607151000.pkl" not in remaining_files
        assert f"lgbm_{symbol}_{tf}_backup_202607151005.pkl" not in remaining_files

        # Свежие файлы должны остаться
        for ts in ["202607151010", "202607151015", "202607151020", "202607151025", "202607151030"]:
            assert f"lgbm_{symbol}_{tf}_backup_{ts}.pkl" in remaining_files