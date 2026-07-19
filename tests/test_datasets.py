import os
import tempfile
import pytest
import pandas as pd
import json

from src.crud.kline import KlineRepository
from src.datasets.build import build_and_save_dataset


@pytest.mark.asyncio
async def test_build_and_save_dataset_success(temp_db_session):
    # 1. Заполняем временную базу данных тестовыми свечами
    repo = KlineRepository(temp_db_session)
    klines_data = []

    # Создадим 35 последовательных часовых свечей
    start_time = 1672531200000  # 2023-01-01 00:00:00 UTC
    for i in range(35):
        klines_data.append(
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "open_time": start_time + i * 3600 * 1000,
                "open": 16800.0 + i * 10,
                "high": 16900.0 + i * 10,
                "low": 16700.0 + i * 10,
                "close": 16850.0 + i * 10,
                "volume": 1000.0 + i * 50,
            }
        )

    await repo.save_klines(klines_data)

    # 2. Запускаем сборку датасета во временную директорию
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = await build_and_save_dataset(
            session=temp_db_session,
            symbol="BTC/USDT",
            timeframe="1h",
            version="1.0",
            horizon=3,
            threshold=0.005,
            output_dir=tmpdir,
        )

        # Проверяем, что файлы действительно созданы на диске
        assert os.path.exists(parquet_path)

        json_path = parquet_path.replace(".parquet", ".json")
        assert os.path.exists(json_path)

        # 3. Считываем данные назад и проверяем колонки
        df_loaded = pd.read_parquet(parquet_path)
        assert "open" in df_loaded.columns
        assert "rsi" in df_loaded.columns
        assert "macd" in df_loaded.columns
        assert "target_binary" in df_loaded.columns
        assert "target_triple" in df_loaded.columns

        # Так как горизонт = 3, последние 3 строки целей должны остаться NaN
        from src.labels.generator import MAX_ADAPTIVE_HORIZON_CANDLES

        assert (
            df_loaded["target_binary"].iloc[-MAX_ADAPTIVE_HORIZON_CANDLES:].isna().all()
        )

        assert not pd.isna(
            df_loaded["target_binary"].iloc[-(MAX_ADAPTIVE_HORIZON_CANDLES + 1)]
        )

        # 4. Проверяем корректность паспорта метаданных
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["symbol"] == "BTC/USDT"
        assert meta["timeframe"] == "1h"
        assert meta["version"] == "1.0"
        assert "target_binary" in meta["targets"]
        assert meta["total_rows"] == 35
        assert "git_sha" in meta

@pytest.mark.asyncio
async def test_build_and_save_dataset_atr_mode_records_metadata(temp_db_session):
    repo = KlineRepository(temp_db_session)
    klines_data = []
    start_time = 1672531200000
    n = 150
    for i in range(n):
        klines_data.append({
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "open_time": start_time + i * 3600000,
            "open": 100.0 + i * 0.1, "high": 101.0 + i * 0.1,
            "low": 99.0 + i * 0.1, "close": 100.5 + i * 0.1,
            "volume": 1000.0,
        })
    await repo.save_klines(klines_data)

    parquet_path = await build_and_save_dataset(
        temp_db_session, symbol="BTC/USDT", timeframe="1h", version="atr-test",
        horizon=5, threshold=0.01, tp_atr_mult=1.5, sl_atr_mult=1.0,
    )
    json_path = parquet_path.replace(".parquet", ".json")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["targets"]["target_triple"]["label_mode"] == "atr"
    assert meta["targets"]["target_triple"]["tp_atr_mult"] == 1.5
    assert meta["targets"]["target_triple"]["sl_atr_mult"] == 1.0