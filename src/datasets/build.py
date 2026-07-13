import os
import json
from datetime import datetime
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
import subprocess
from loguru import logger

from src.crud.kline import KlineRepository
from src.features.engineering import add_features
from src.labels.generator import generate_binary_labels, generate_triple_labels


def get_git_sha() -> str:
    """
    Получает хэш текущего коммита Git.
    Если утилита Git недоступна в системе, возвращает 'unknown'.
    """
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
    """
    Скачивает свечи из БД, рассчитывает все признаки и цели,
    сохраняет готовую таблицу в Parquet и описание в JSON.
    Возвращает путь к созданному файлу данных.
    """
    # 1. Загружаем свечи из базы данных
    repo = KlineRepository(session)
    # Берем с запасом последние 10 000 свечей
    klines = await repo.get_klines(symbol, timeframe, limit=10000)

    if not klines:
        raise ValueError(
            f"В базе данных нет свечей для пары {symbol} и таймфрейма {timeframe}."
        )

    # Превращаем данные из базы в таблицу pandas
    # get_klines отдает свечи от новых к старым (desc),
    # для расчета индикаторов нам нужно отсортировать их от старых к новым (asc)
    data = []
    for k in klines:
        data.append(
            {
                "open_time": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
        )
    df = pd.DataFrame(data).sort_values("open_time").reset_index(drop=True)

    # 2. Запускаем расчет индикаторов (RSI, MACD, волатильность...)
    df_features = add_features(df)

    # 3. Рассчитываем два типа целей (бинарную и тройную) для универсальности датасета
    df_features["target_binary"] = generate_binary_labels(
        df_features, horizon=horizon, threshold=0.0
    )
    df_features["target_triple"] = generate_triple_labels(
        df_features, horizon=horizon, threshold=threshold
    )

    # 4. Определяем временные рамки нашего датасета
    first_time_ms = int(df_features["open_time"].iloc[0])
    last_time_ms = int(df_features["open_time"].iloc[-1])

    start_date = datetime.fromtimestamp(first_time_ms / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    end_date = datetime.fromtimestamp(last_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")

    # 5. Подготавливаем имена файлов (например, убираем косую черту: BTC/USDT -> BTCUSDT)
    clean_symbol = symbol.replace("/", "").replace(":", "")

    os.makedirs(output_dir, exist_ok=True)

    parquet_filename = f"{clean_symbol}_{timeframe}_v{version}.parquet"
    json_filename = f"{clean_symbol}_{timeframe}_v{version}.json"

    parquet_path = os.path.join(output_dir, parquet_filename)
    json_path = os.path.join(output_dir, json_filename)

    # 6. Сохраняем готовую таблицу в файл Parquet
    df_features.to_parquet(parquet_path, index=False)

    # 7. Заполняем и сохраняем паспорт данных в JSON
    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "version": version,
        "features": [
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "volatility",
            "volume_ratio",
        ],
        "targets": {
            "target_binary": {"type": "binary", "horizon": horizon, "threshold": 0.0},
            "target_triple": {
                "type": "triple",
                "horizon": horizon,
                "threshold": threshold,
            },
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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"Датасет успешно собран и записан в {parquet_path}")
    return parquet_path