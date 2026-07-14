import pytest
import pandas as pd
import numpy as np

from src.strategy.signals import simulate_strategy


def test_simulate_strategy_time_exit():
    # 20 свечей с постепенным ростом цены
    df = pd.DataFrame(
        {
            "close": [100.0 + i for i in range(20)],
            "high": [100.0 + i + 0.5 for i in range(20)],
            "low": [100.0 + i - 0.5 for i in range(20)],
            # Входим на свече с индексом 2 (цена close 102.0)
            "predicted_signal": [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    )

    # Симулируем удержание 3 свечи, без SL/TP (выход только по времени)
    metrics = simulate_strategy(
        df,
        predicted_col="predicted_signal",
        horizon=3,
        sl_pct=None,
        tp_pct=None,
        transaction_cost=0.001,
    )

    # Должна совершиться ровно 1 сделка
    assert metrics["total_trades"] == 1

    # Вход по close[2] = 102. Выход по close[5] (через 3 свечи) = 105.
    # Доходность: (105 - 102) / 102 = 2.94117% (0.0294117)
    # Комиссия: 2 * 0.1% = 0.2% (0.002)
    # Итог: ~0.0274117
    expected_return = (105 - 102) / 102 - (2 * 0.001)

    assert pytest.approx(metrics["total_return"], abs=1e-5) == expected_return
    assert metrics["win_rate"] == 1.0
    assert metrics["expectancy"] > 0


def test_simulate_strategy_sl_hit():
    # Имитируем падение для проверки срабатывания Stop-Loss
    df = pd.DataFrame(
        {
            "close": [100, 101, 102, 95, 96, 97],
            "high": [100.5, 101.5, 102.5, 95.5, 96.5, 97.5],
            # На свече 3 (индекс 3) low обвалился до 90.0
            "low": [99.5, 100.5, 101.5, 90.0, 95.5, 96.5],
            # Входим по close[2] = 102.0
            "predicted_signal": [0, 0, 1, 0, 0, 0],
        }
    )

    # Задаем SL = 2% (0.02). Уровень сработки: 102 * 0.98 = 99.96
    # На свече 3 (low = 90.0) уровень 99.96 будет гарантированно пробит
    metrics = simulate_strategy(
        df,
        predicted_col="predicted_signal",
        horizon=3,
        sl_pct=0.02,
        tp_pct=None,
        transaction_cost=0.0,
    )

    assert metrics["total_trades"] == 1
    # Доходность должна составить ровно -2% (минус 0.02)
    assert pytest.approx(metrics["total_return"], abs=1e-5) == -0.02


def test_simulate_strategy_no_signals():
    # Сигналов нет — сделок быть не должно
    df = pd.DataFrame(
        {
            "close": [100, 101, 102],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "predicted_signal": [0, 0, 0],
        }
    )

    metrics = simulate_strategy(df)
    assert metrics["total_trades"] == 0
    assert metrics["total_return"] == 0.0


def test_simulate_strategy_short_tp_hit():
    # Имитируем падение цены для проверки SHORT сделки
    df = pd.DataFrame(
        {
            "close": [100.0, 99.0, 98.0, 95.0, 96.0],
            "high": [100.5, 99.5, 98.5, 95.5, 96.5],
            "low": [99.5, 98.5, 97.5, 94.0, 95.5],
            # Входим в SHORT на свече с индексом 1 (close = 99.0)
            "predicted_signal": [0, -1, 0, 0, 0],
        }
    )

    # Задаем TP = 3% (0.03). Уровень сработки для шорта: 99.0 * (1 - 0.03) = 96.03
    # На свече 3 (index 3) low обвалился до 94.0, что пробивает наш TP (94.0 <= 96.03)
    metrics = simulate_strategy(
        df,
        predicted_col="predicted_signal",
        horizon=3,
        sl_pct=0.02,
        tp_pct=0.03,
        transaction_cost=0.0,
    )

    assert metrics["total_trades"] == 1
    # Должна зафиксироваться прибыль ровно 3% (0.03)
    assert pytest.approx(metrics["total_return"], abs=1e-5) == 0.03
    assert metrics["win_rate"] == 1.0