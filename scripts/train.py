import argparse
import asyncio
import os
import tempfile
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select

from src.config import get_settings
from src.db import AsyncSessionFactory, Kline, init_db
from src.strategy.features import add_features
from src.strategy.model import EconomicReturnRegressor

FEATURE_COLS = [
    "rsi", "macd_pct", "macd_signal_pct", "macd_hist_pct",
    "bb_upper_pct", "bb_lower_pct", "atr_pct", "adx",
    "volatility", "volume_ratio", "return_1", "return_3"
]
def calculate_post_cost_targets(
    df: pd.DataFrame,
    horizon: int,
    sl_pct: float,
    tp_pct: float,
    commission_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Рассчитывает доходности с комиссиями; при касании обоих уровней выбирает SL."""
    long_returns: list[float] = []
    short_returns: list[float] = []

    for idx in range(len(df) - horizon):
        entry_price = float(df.iloc[idx + 1]["open"])
        candles = df.iloc[idx + 1 : idx + 1 + horizon]
        long_exit = float(candles["close"].iloc[-1])
        short_exit = long_exit

        for _, candle in candles.iterrows():
            if candle["low"] <= entry_price * (1 - sl_pct):
                long_exit = entry_price * (1 - sl_pct)
                break
            if candle["high"] >= entry_price * (1 + tp_pct):
                long_exit = entry_price * (1 + tp_pct)
                break

        for _, candle in candles.iterrows():
            if candle["high"] >= entry_price * (1 + sl_pct):
                short_exit = entry_price * (1 + sl_pct)
                break
            if candle["low"] <= entry_price * (1 - tp_pct):
                short_exit = entry_price * (1 - tp_pct)
                break

        long_returns.append(
            (long_exit * (1 - commission_rate) - entry_price * (1 + commission_rate))
            / entry_price
        )
        short_returns.append(
            (entry_price * (1 - commission_rate) - short_exit * (1 + commission_rate))
            / entry_price
        )

    return np.asarray(long_returns, dtype=np.float32), np.asarray(
        short_returns, dtype=np.float32
    )


async def train_model(symbol: str, timeframe: str):
    await init_db()
    settings = get_settings()

    async with AsyncSessionFactory() as session:
        res = await session.execute(
            select(Kline).where(Kline.symbol == symbol, Kline.timeframe == timeframe).order_by(Kline.open_time.asc())
        )
        klines = res.scalars().all()

        if len(klines) < 300:
            logger.error(f"Недостаточно свечей ({len(klines)} < 300). Запустите python -m scripts.backfill")
            return

        df = pd.DataFrame([{
            "open_time": k.open_time, "open": k.open, "high": k.high,
            "low": k.low, "close": k.close, "volume": k.volume
        } for k in klines])

    # 1. Расчет фичей
    df_feat = add_features(df)

    # 2. Создание post-cost таргетов
    horizon = settings.DEFAULT_TIMEOUT_CANDLES
    tp_pct = settings.DEFAULT_TP_PCT
    sl_pct = settings.DEFAULT_SL_PCT

    long_returns, short_returns = calculate_post_cost_targets(
        df_feat, horizon, sl_pct, tp_pct, settings.DEFAULT_COMMISSION_RATE
    )

    df_train = df_feat.iloc[:len(long_returns)].copy()
    df_train["long_target"] = long_returns
    df_train["short_target"] = short_returns

    df_clean = df_train.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # 3. Обучение LightGBM
    model = EconomicReturnRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
    model.fit(df_clean[FEATURE_COLS], df_clean["long_target"], df_clean["short_target"])

    clean_sym = symbol.replace("/", "").replace(":", "")
    os.makedirs("models/saved_models", exist_ok=True)
    model_path = f"models/saved_models/lgbm_{clean_sym}_{timeframe}.pkl"

    artifact = {
        "model": model,
        "features": FEATURE_COLS,
        "min_expected_return": settings.MIN_EXPECTED_RETURN,
        "symbol": symbol,
        "timeframe": timeframe
    }

    with tempfile.NamedTemporaryFile(dir="models/saved_models", delete=False) as temp_file:
        pickle.dump(artifact, temp_file)
        temp_path = temp_file.name
    os.replace(temp_path, model_path)

    logger.info(f"Модель сохранена: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    asyncio.run(train_model(args.symbol, args.timeframe))
