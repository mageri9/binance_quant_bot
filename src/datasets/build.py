import os
import json
import asyncio
from datetime import datetime
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
import subprocess
from loguru import logger

from src.crud.kline import KlineRepository
from src.utils.memory import downcast_dtypes
from src.features.engineering import add_features
from src.labels.generator import generate_binary_labels, generate_triple_labels


def get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return sha.decode("utf-8").strip()
    except Exception:
        return "unknown"


async def build_and_save_dataset(
    session: AsyncSession,
    symbol: str,
    timeframe: str,
    version: str,
    horizon: int = 5,
    threshold: float = 0.01,
    output_dir: str = "datasets",
) -> str:
    repo = KlineRepository(session)
    klines = await repo.get_klines(symbol, timeframe, limit=20000)

    if not klines:
        raise ValueError(
            f"В базе данных нет свечей для пары {symbol} и таймфрейма {timeframe}."
        )

    data = [
        {
            "open_time": k.open_time,
            "open": k.open,
            "high": k.high,
            "low": k.low,
            "close": k.close,
            "volume": k.volume,
        }
        for k in klines
    ]
    df = pd.DataFrame(data).sort_values("open_time").reset_index(drop=True)

    clean_symbol = symbol.replace("/", "").replace(":", "")
    os.makedirs(output_dir, exist_ok=True)

    parquet_path = os.path.join(output_dir, f"{clean_symbol}_{timeframe}_v{version}.parquet")
    json_path = os.path.join(output_dir, f"{clean_symbol}_{timeframe}_v{version}.json")

    # Переносим расчет признаков, меток и сохранение Parquet в фоновый поток
    def process_data_sync() -> pd.DataFrame:
        df_feats = add_features(df)
        df_feats["target_binary"] = generate_binary_labels(df_feats, horizon=horizon, threshold=0.0)
        df_feats["target_triple"] = generate_triple_labels(df_feats, horizon=horizon, threshold=threshold)
        df_feats = downcast_dtypes(df_feats)
        df_feats.to_parquet(parquet_path, index=False)
        return df_feats

    df_features = await asyncio.to_thread(process_data_sync)

    first_time_ms = int(df_features["open_time"].iloc[0])
    last_time_ms = int(df_features["open_time"].iloc[-1])
    start_date = datetime.fromtimestamp(first_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.fromtimestamp(last_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")

    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "version": version,
        "features": [
            "rsi", "macd_pct", "macd_signal_pct", "macd_hist_pct", "volatility",
            "volume_ratio", "bb_upper_pct", "bb_middle_pct", "bb_lower_pct", "atr_pct", "adx"
        ],
        "targets": {
            "target_binary": {"type": "binary", "horizon": horizon, "threshold": 0.0},
            "target_triple": {"type": "triple", "horizon": horizon, "threshold": threshold},
        },
        "date_range": {
            "start_time_ms": first_time_ms,
            "end_time_ms": last_time_ms,
            "start_date": start_date,
            "end_date": end_date,
        },
        "total_rows": len(df_features),
        "git_sha": get_git_sha(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    def save_metadata_sync():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    await asyncio.to_thread(save_metadata_sync)

    logger.info(f"Датасет успешно собран и записан в {parquet_path}")
    return parquet_path