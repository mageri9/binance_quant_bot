from typing import Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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

    # Database & Redis
    DATABASE_URL: str = "sqlite+aiosqlite:///./marketmind.db"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # Risk Management
    RISK_MAX_ALLOCATION_PCT: float = 0.10
    RISK_MAX_DAILY_LOSS_PCT: float = 0.05
    RISK_CONSECUTIVE_LOSSES_LIMIT: int = 5

    # Strategy Defaults
    DEFAULT_SL_PCT: float = 0.02
    DEFAULT_TP_PCT: float = 0.04
    DEFAULT_TIMEOUT_CANDLES: int = 5
    MIN_EXPECTED_RETURN: float = 0.001

    # Торговые пары и таймфреймы
    ACTIVE_CONFIGS: list[tuple[str, str]] = [
        ("BTC/USDT", "1h"),
        ("ETH/USDT", "1h"),
        ("SOL/USDT", "1h"),
    ]

    @field_validator("ADMIN_IDS", mode="before")
    @classmethod
    def parse_admin_ids(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()