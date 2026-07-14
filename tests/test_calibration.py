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

    results = perform_grid_search(df_valid, sl_grid, tp_grid, horizon=3)

    assert len(results) == 1
    best_res = results[0]
    assert best_res["sl_pct"] == 0.02
    assert best_res["tp_pct"] == 0.029
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
        mock_settings.LABEL_HORIZON = 5

        # perform_grid_search мокаем, чтобы изолированно проверить только
        # декодирование сигнала, не завися от исхода реальной симуляции сделок
        fake_grid_result = [{
            "sl_pct": 0.02, "tp_pct": 0.04, "total_trades": 1,
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
            sl, tp, report = await get_best_calibration("BTC/USDT", "1h")

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