import pandas as pd
import pytest
from scripts.calibrate import perform_grid_search


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