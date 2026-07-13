from pathlib import Path
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings


def get_env_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    # Bot
    BOT_TOKEN: str
    ADMIN_IDS: list[int]

    # Model
    MODEL_PATH: str = "models/saved_models/lgbm_BTCUSDT_1h.pkl"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # Rate limiting
    RATE_LIMIT_CALLS: int = 5
    RATE_LIMIT_PERIOD: int = 10  # seconds

    # Logging
    LOG_LEVEL: str = "INFO"

    # Retraining
    RETRAIN_INTERVAL_SECONDS: int = 86400  # раз в сутки
    MIN_KLINES_FOR_TRAIN: int = 1200  # train_size + test_size с запасом
    TRAIN_SIZE: int = 1000
    TEST_SIZE: int = 200
    LABEL_HORIZON: int = 5
    LABEL_THRESHOLD: float = 0.01
    TARGET_COL: str = "target_triple"

    # Paper Trading Defaults
    PAPER_SL_PCT: float = 0.02
    PAPER_TP_PCT: float = 0.04

    # Paper Trading Position Sizing
    PAPER_RISK_PCT: float = 0.10
    PAPER_MIN_ALLOCATION: float = 1.0

    class Config:
        env_file = get_env_path()
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"   # добавлено: Nexus SRE переменные читаются напрямую через os.getenv в main.py

    @field_validator("ADMIN_IDS", mode="before")
    @classmethod
    def parse_admin_ids(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @property
    def db_url(self) -> str:
        db_path = Path(__file__).resolve().parent.parent / "db" / "db.db"
        return f"sqlite+aiosqlite:///{db_path}"

    @property
    def db_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "db" / "db.db"

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
