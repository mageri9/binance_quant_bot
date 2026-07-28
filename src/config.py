import functools
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram Bot
    BOT_TOKEN: str
    ADMIN_IDS: list[int]

    # Trading Mode: paper | testnet | mainnet
    TRADING_MODE: Literal["paper", "testnet", "mainnet"] = "testnet"

    # Binance API
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_PROXY: str = ""

    # Nexus SRE (Ваш SDK)
    NEXUS_APP_SECRET: str = ""
    NEXUS_ENDPOINT_URL: str = "http://nexus-webhook:8000/events/app"
    NEXUS_PROJECT_NAME: str = "binance_quant_bot"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./marketmind.db"

    # Risk Management
    RISK_MAX_ALLOCATION_PCT: float = 0.10
    RISK_MAX_DAILY_LOSS_PCT: float = 0.05
    RISK_CONSECUTIVE_LOSSES_LIMIT: int = 5

    # Strategy Defaults
    DEFAULT_SL_PCT: float = 0.02
    DEFAULT_TP_PCT: float = 0.04
    DEFAULT_TIMEOUT_CANDLES: int = 5
    DEFAULT_COMMISSION_RATE: float = 0.0004
    MIN_EXPECTED_RETURN: float = 0.0025

    # Periodic Telegram digest
    DIGEST_INTERVAL_SECONDS: int = 86400

    # Торговые пары и таймфреймы
    ACTIVE_CONFIGS: list[tuple[str, str]] = [
        ("BTC/USDT", "1h"),
    ]

    @field_validator("DATABASE_URL")
    @classmethod
    def require_sqlite_database(cls, value: str) -> str:
        if not value.startswith("sqlite+aiosqlite://"):
            raise ValueError("DATABASE_URL must use the sqlite+aiosqlite driver")
        return value

    @field_validator("DIGEST_INTERVAL_SECONDS")
    @classmethod
    def require_positive_digest_interval(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("DIGEST_INTERVAL_SECONDS must be positive")
        return value


@functools.lru_cache()
def get_settings() -> Settings:
    return Settings()
