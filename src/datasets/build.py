import os
import json
import asyncio
from datetime import datetime
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
import subprocess
import hashlib
from loguru import logger

from src.crud.kline import KlineRepository
from src.utils.memory import downcast_dtypes
from src.features.engineering import DEFAULT_FEATURE_SCHEMA, add_features
from src.execution.trade import TradePolicy, build_trade_targets
from src.execution.kernel import ExecutionCosts


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
    tp_atr_mult: float | None = None,
    sl_atr_mult: float | None = None,
    trade_policy: TradePolicy | None = None,
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

    def process_data_sync() -> pd.DataFrame:
        df_feats = add_features(df)
        policy = trade_policy or TradePolicy(
            timeout_candles=horizon, sl_pct=threshold, tp_pct=threshold,
            costs=ExecutionCosts(commission_rate=0, slippage_rate=0,
                                 bid_ask_spread_rate=0, funding_rate_per_trade=0),
        )
        targets = build_trade_targets(df_feats, policy)
        # Legacy callers historically reserved the adaptive-ATR maximum tail.
        # Production supplies trade_policy and reserves its exact fixed timeout.
        if trade_policy is None:
            from src.labels.generator import MAX_ADAPTIVE_HORIZON_CANDLES
            targets.iloc[-MAX_ADAPTIVE_HORIZON_CANDLES:] = pd.NA
        for column in targets:
            df_feats[column] = targets[column]
        df_feats = downcast_dtypes(df_feats)
        return df_feats

    df_features = await asyncio.to_thread(process_data_sync)

    first_time_ms = int(df_features["open_time"].iloc[0])
    last_time_ms = int(df_features["open_time"].iloc[-1])
    start_date = datetime.fromtimestamp(first_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.fromtimestamp(last_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")

    feature_schema = DEFAULT_FEATURE_SCHEMA
    effective_policy = trade_policy or TradePolicy(
        timeout_candles=horizon, sl_pct=threshold, tp_pct=threshold,
        costs=ExecutionCosts(commission_rate=0, slippage_rate=0,
                             bid_ask_spread_rate=0, funding_rate_per_trade=0),
    )
    label_config = {
        "horizon": horizon, "threshold": threshold,
        "tp_atr_mult": tp_atr_mult, "sl_atr_mult": sl_atr_mult,
        "trade_policy": effective_policy.identity(),
    }
    candle_bytes = pd.util.hash_pandas_object(
        df[["open_time", "open", "high", "low", "close", "volume"]], index=False
    ).values.tobytes()
    identity = json.dumps({
        "symbol": symbol, "timeframe": timeframe,
        "range": [first_time_ms, last_time_ms],
        "feature_schema": feature_schema, "label_config": label_config,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    dataset_fingerprint = hashlib.sha256(candle_bytes + identity).hexdigest()
    version = dataset_fingerprint[:16]
    clean_symbol = symbol.replace("/", "").replace(":", "")
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, f"{clean_symbol}_{timeframe}_v{version}.parquet")
    json_path = os.path.join(output_dir, f"{clean_symbol}_{timeframe}_v{version}.json")
    await asyncio.to_thread(df_features.to_parquet, parquet_path, index=False)

    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "version": version,
        "dataset_fingerprint": dataset_fingerprint,
        "feature_schema_hash": hashlib.sha256(json.dumps(feature_schema).encode()).hexdigest(),
        "label_config_hash": hashlib.sha256(json.dumps(label_config, sort_keys=True).encode()).hexdigest(),
        "features": feature_schema,
        "targets": {
            "target_binary": {
                "type": "binary", "horizon": horizon, "threshold": 0.0,
                "label_mode": "trade_policy" if trade_policy is not None else (
                    "atr" if tp_atr_mult is not None else "fixed_threshold"
                ),
                "tp_atr_mult": tp_atr_mult,
            },
            "target_triple": {
                "type": "triple", "horizon": horizon, "threshold": threshold,
                "label_mode": "trade_policy" if trade_policy is not None else (
                    "atr" if tp_atr_mult is not None and sl_atr_mult is not None else "fixed_threshold"
                ),
                "tp_atr_mult": tp_atr_mult,
                "sl_atr_mult": sl_atr_mult,
            },
        },
        "trade_policy": effective_policy.identity(),
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
