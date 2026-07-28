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

from src.config import Settings, get_settings
from src.db import AsyncSessionFactory, Kline, init_db
from src.strategy.features import add_features
from src.strategy.model import EconomicReturnRegressor

FEATURE_COLS = [
    "rsi", "macd_pct", "macd_signal_pct", "macd_hist_pct",
    "bb_upper_pct", "bb_middle_pct", "bb_lower_pct", "atr_pct", "adx",
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


def evaluate_oos(
    model: EconomicReturnRegressor, test_df: pd.DataFrame, min_expected_return: float
) -> tuple[int, float]:
    """Return the number and realized mean return of threshold-passing OOS signals."""
    pred_long, pred_short = model.predict_returns(test_df[FEATURE_COLS])
    pred_long = np.asarray(pred_long)
    pred_short = np.asarray(pred_short)
    long_returns = test_df.loc[
        (pred_long > pred_short) & (pred_long >= min_expected_return), "long_target"
    ].to_numpy()
    short_returns = test_df.loc[
        (pred_short > pred_long) & (pred_short >= min_expected_return), "short_target"
    ].to_numpy()
    realized_returns = np.concatenate((long_returns, short_returns))

    n_trades = len(realized_returns)
    mean_return = float(realized_returns.mean()) if n_trades else 0.0
    return n_trades, mean_return


def save_model_artifact(artifact: dict, model_path: str) -> None:
    """Atomically replace the active artifact only after it passed validation."""
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=model_dir, delete=False) as temp_file:
            temp_path = temp_file.name
            pickle.dump(artifact, temp_file)
        os.replace(temp_path, model_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def train_and_save_model(
    df_clean: pd.DataFrame, settings: Settings, symbol: str, timeframe: str, horizon: int
) -> bool:
    """Train, validate on the chronological OOS tail, and save only approved models."""
    split_idx = int(len(df_clean) * 0.7)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    purged_train_df = train_df.iloc[:-horizon] if horizon else train_df

    if purged_train_df.empty or test_df.empty:
        logger.warning("Quality Gate FAILED: insufficient data after chronological split.")
        return False

    model = EconomicReturnRegressor(
        n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1
    )
    model.fit(
        purged_train_df[FEATURE_COLS],
        purged_train_df["long_target"],
        purged_train_df["short_target"],
    )
    n_trades, mean_return = evaluate_oos(
        model, test_df, settings.MIN_EXPECTED_RETURN
    )
    if mean_return <= 0 or n_trades < 30:
        logger.warning(
            f"Quality Gate FAILED: n_trades={n_trades} (min 30), "
            f"mean_return={mean_return:.5f} (min > 0). Модель не сохранена."
        )
        return False

    # Refit on every clean row only after the independent OOS gate succeeds.
    model.fit(df_clean[FEATURE_COLS], df_clean["long_target"], df_clean["short_target"])
    clean_sym = symbol.replace("/", "").replace(":", "")
    model_path = f"models/saved_models/lgbm_{clean_sym}_{timeframe}.pkl"
    artifact = {
        "model": model,
        "features": FEATURE_COLS,
        "min_expected_return": settings.MIN_EXPECTED_RETURN,
        "symbol": symbol,
        "timeframe": timeframe,
    }
    save_model_artifact(artifact, model_path)
    logger.info(
        f"Quality Gate PASSED: n_trades={n_trades}, mean_return={mean_return:.5f}"
    )
    return True


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
    train_and_save_model(df_clean, settings, symbol, timeframe, horizon)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    asyncio.run(train_model(args.symbol, args.timeframe))
