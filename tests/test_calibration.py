import pandas as pd
import pytest
from scripts.calibrate import perform_grid_search
import os
import pickle
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from src.core.db import Base
from src.crud.kline import KlineRepository
from scripts.calibrate import get_best_calibration
from src.utils.artifact_paths import get_oos_path

def test_perform_grid_search_success():
    # Создадим фиктивную валидную выборку, где сигнал на покупку возникает на индексе 2 (вход по 102)
    # Имитируем рост, чтобы сработал Take-Profit на значении 105
    df_valid = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5, 104.5],
            "predicted_signal": [0, 0, 1, 0, 0, 0],
        }
    )

    sl_grid = [0.02]
    tp_grid = [0.029]  # (105 - 102) / 102 ≈ 2.94% (при tp=2.9% сработает TP на 102 * 1.029 = 104.958, что ниже high=105.5)

    results = perform_grid_search(df_valid, sl_grid, tp_grid, horizon_grid=[3], min_trades=1)

    assert len(results) == 1
    best_res = results[0]
    assert best_res["sl_pct"] == 0.02
    assert best_res["tp_pct"] == 0.029
    assert best_res["horizon"] == 3
    assert best_res["total_trades"] == 1
    assert best_res["win_rate"] == 1.0


class FakeTripleModel:
    """
    Заглушка модели для target_triple.
    Возвращает сырые (замапленные) классы {0, 1, 2}, как это делает
    настоящий LGBMClassifier после обучения на target_col="target_triple"
    """

    def predict(self, X):
        n = len(X)
        return np.arange(n) % 3


@pytest.mark.asyncio
async def test_get_best_calibration_decodes_triple_model_classes():
    """
    Проверяем, что predicted_signal, попадающий в perform_grid_search,
    содержит только декодированные значения {-1.0, 0.0, 1.0}.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Использование в памяти СУБД с StaticPool позволяет держать базу активной
        # между разными сессиями и полностью решает проблему блокировки файлов в Windows
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )

        # 2. Заполняем свечи (>=100, как требует get_best_calibration)
        n_rows = 150
        np.random.seed(42)
        klines_data = []
        for i in range(n_rows):
            klines_data.append({
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "open_time": 1000 + i * 3600 * 1000,
                "open": float(np.random.uniform(100, 110)),
                "high": float(np.random.uniform(110, 120)),
                "low": float(np.random.uniform(90, 100)),
                "close": float(np.random.uniform(100, 110)),
                "volume": float(np.random.uniform(1000, 5000)),
            })

        async with session_factory() as session:
            repo = KlineRepository(session)
            await repo.save_klines(klines_data)

        # 3. Готовим псевдо-модель, обученную на target_triple
        feature_cols = ["rsi", "macd", "macd_signal", "macd_hist", "volatility", "volume_ratio"]
        model_path = os.path.join(tmpdir, "fake_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": FakeTripleModel(),
                "features": feature_cols,
                "scaler": None,
                "target_col": "target_triple",
            }, f)

        mock_settings = MagicMock()
        mock_settings.MODEL_PATH = model_path
        # Обучаем мок возвращать реальный путь к нашему временному файлу pickle
        mock_settings.get_model_path.return_value = model_path
        mock_settings.LABEL_HORIZON = 5
        mock_settings.CALIBRATION_MIN_TRADES = 1

        # perform_grid_search мокаем, чтобы изолированно проверить только
        # декодирование сигнала, не завися от исхода реальной симуляции сделок
        fake_grid_result = [{
            "sl_pct": 0.02, "tp_pct": 0.04, "horizon": 5, "total_trades": 1,
            "win_rate": 1.0, "profit_factor": 2.0, "sharpe_ratio": 1.5,
            "sortino_ratio": 1.2, "expectancy": 0.01, "total_return": 0.05,
        }]

        with (
            patch("scripts.calibrate.get_settings", return_value=mock_settings),
            patch("scripts.calibrate.AsyncSessionFactory", session_factory),
            patch(
                "scripts.calibrate.perform_grid_search",
                return_value=fake_grid_result,
            ) as mock_grid_search,
        ):
            sl, tp, hz, report, honest_metrics = await get_best_calibration("BTC/USDT", "1h")

        await engine.dispose()

        # 4. Проверяем, что в perform_grid_search попал декодированный сигнал
        assert mock_grid_search.called
        captured_df = mock_grid_search.call_args[0][0]

        assert "predicted_signal" in captured_df.columns

        assert set(captured_df["predicted_signal"].unique()).issubset({-1.0, 0.0, 1.0})

        expected = pd.Series(
            np.arange(len(captured_df)) % 3,
            index=captured_df.index,
        ).map({0: -1.0, 1: 0.0, 2: 1.0})

        pd.testing.assert_series_equal(
            captured_df["predicted_signal"].astype(float),
            expected.astype(float),
            check_names=False,
        )

        assert sl == 0.02
        assert tp == 0.04
        assert hz == 5

def test_perform_grid_search_filters_low_trade_count():
    df_valid = pd.DataFrame({
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "high": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        "low": [99.5, 100.5, 101.5, 102.5, 103.5, 104.5],
        "predicted_signal": [0, 0, 1, 0, 0, 0],
    })
    results = perform_grid_search(
        df_valid, sl_grid=[0.02], tp_grid=[0.029], horizon_grid=[3], min_trades=5,
    )
    assert results == []

@pytest.mark.asyncio
async def test_get_best_calibration_prefers_sibling_oos_parquet(tmp_path):
    """
    Если рядом с моделью лежит {model}_oos.parquet, get_best_calibration
    обязан использовать его и не должен трогать БД вообще.
    """
    from src.utils.artifact_paths import get_oos_path

    model_path = str(tmp_path / "lgbm_BTCUSDT_1h.pkl")
    oos_path = get_oos_path(model_path)

    # Строим OOS-датафрейм с гарантированным TP-сценарием, как в
    # test_perform_grid_search_success
    df_oos = pd.DataFrame({
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "high": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        "low": [99.5, 100.5, 101.5, 102.5, 103.5, 104.5],
        "predicted_signal": [0, 0, 1, 0, 0, 0],
    })
    df_oos.to_parquet(oos_path, index=False)

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": "dummy",
            "features": ["rsi"],
            "scaler": None,
            "target_col": "target_binary",
        }, f)

    mock_settings = MagicMock()
    mock_settings.get_model_path.return_value = model_path
    mock_settings.CALIBRATION_MIN_TRADES = 1

    with (
        patch("scripts.calibrate.get_settings", return_value=mock_settings),
        patch("src.crud.kline.KlineRepository.get_klines") as mock_get_klines,
    ):
        sl, tp, hz, report, honest_metrics = await get_best_calibration("BTC/USDT", "1h", custom_model_path=model_path)

    # БД не должна была вызываться вообще — данные пришли из parquet
    mock_get_klines.assert_not_called()
    assert sl is not None
    assert tp is not None

def test_perform_grid_search_atr_mode():
    df_valid = pd.DataFrame({
        "close": [100.0, 100.0, 102.5, 102.5, 100.0] * 5,
        "high":  [100.0, 100.0, 102.5, 102.5, 100.0] * 5,
        "low":   [100.0, 100.0, 101.5, 101.5, 100.0] * 5,
        "atr":   [1.0] * 25,
        "predicted_signal": [1, 0, 0, 0, 0] * 5,
    })
    results = perform_grid_search(
        df_valid, k_sl_grid=[1.0], k_tp_grid=[2.0], horizon_grid=[3], min_trades=1,
    )
    assert len(results) >= 1
    assert "sl_atr_mult" in results[0]
    assert "tp_atr_mult" in results[0]
    assert "sl_pct" not in results[0]

@pytest.mark.asyncio
async def test_get_best_calibration_applies_meta_gate(tmp_path):
    class RejectAllMeta:
        classes_ = [0, 1]
        def predict_proba(self, X):
            return np.tile([1.0, 0.0], (len(X), 1))  # всегда "низкая вероятность успеха"

    model_path = str(tmp_path / "lgbm_BTCUSDT_1h.pkl")
    oos_path = get_oos_path(model_path)

    n = 40
    df_oos = pd.DataFrame({
        "open_time": np.arange(n),
        "close": [100.0] * n, "high": [100.5] * n, "low": [99.5] * n,
        "adx": [25.0] * n, "atr_pct": [0.01] * n, "volume_ratio": [1.0] * n, "volatility": [0.02] * n,
        "predicted_signal": ([1, 0, -1, 0] * (n // 4)),
    })
    df_oos.to_parquet(oos_path, index=False)

    with open(model_path, "wb") as f:
        pickle.dump({"model": None, "features": []}, f)

    with pytest.raises(ValueError):
        # Все сигналы погашены meta-гейтом -> сделок 0 -> min_trades не набирается
        await get_best_calibration(
            "BTC/USDT", "1h",
            custom_model_path=model_path,
            meta_model=RejectAllMeta(),
            meta_features=["adx"],
            meta_threshold=0.5,
        )
