from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


def get_env_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    # Bot
    BOT_TOKEN: str
    ADMIN_IDS: list[int]

    # Database
    DATABASE_URL: str = ""

    # Model
    MODEL_PATH: str = "models/saved_models/lgbm_BTCUSDT_1h.pkl"
    # Deprecated: probabilities are diagnostics, never a trading edge.
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.55
    EDGE_THRESHOLD_SWEEP_ENABLED: bool = True
    EDGE_THRESHOLD_GRID: list[float] = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    EDGE_MIN_COVERAGE: float = 0.20
    OPTUNA_OBJECTIVE_METRIC: Literal["sharpe", "sortino", "expectancy", "profit_factor", "utility"] = "sharpe"
    SOFT_REGIME_ENSEMBLE_ENABLED: bool = True
    SOFT_REGIME_TEMPERATURE: float = 1.0

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # Rate limiting
    RATE_LIMIT_CALLS: int = 5
    RATE_LIMIT_PERIOD: int = 10  # seconds

    # Logging
    LOG_LEVEL: str = "DEBUG"

    # Retraining
    RETRAIN_INTERVAL_SECONDS: int = 21600  # Backward-compatible control interval.
    RETRAIN_POLL_SECONDS: int = 900
    RETRAIN_MIN_NEW_LABELS: int = 200
    RETRAIN_CONTROL_MAX_AGE_HOURS: int = 168
    LIVE_METRIC_MIN_TRADES: int = 30
    LIVE_METRIC_MIN_WIN_RATE: float = 0.40
    MODEL_AUTO_PROMOTE_LEGACY: bool = False
    RECONCILIATION_INTERVAL_SECONDS: int = 30
    MIN_KLINES_FOR_TRAIN: int = 8000
    TRAIN_SIZE: int = 3500
    TEST_SIZE: int = 300
    LABEL_HORIZON: int = 5
    LABEL_THRESHOLD: float = 0.01
    # The model estimates expected, post-cost return for each executable side;
    # it does not learn a barrier-first LONG/SHORT/HOLD class.
    TARGET_COL: str = "expected_return"
    MIN_EXPECTED_RETURN: float = 0.0

    # Paper Trading Defaults
    PAPER_SL_PCT: float = 0.02
    PAPER_TP_PCT: float = 0.04

    # Canonical order contract. Target generation and execution both consume
    # these fields through trade_policy_from_settings().
    TRADE_TIMEOUT_CANDLES: int = 5
    TRADE_SL_PCT: float = 0.02
    TRADE_TP_PCT: float = 0.04

    # Paper Trading Position Sizing
    PAPER_RISK_PCT: float = 0.10
    PAPER_MIN_ALLOCATION: float = 1.0

    # Backtest-only risk experiment; empty values keep fixed 100% sizing.
    BACKTEST_STOP_RISK_PCT: float | None = 0.01
    BACKTEST_TARGET_VOLATILITY: float | None = 0.01
    BACKTEST_MAX_POSITION_PCT: float = 1.0

    # Single Binance Futures execution schedule for backtests and paper fills.
    EXECUTION_COMMISSION: float = 0.0004
    EXECUTION_SLIPPAGE: float = 0.0002
    EXECUTION_BID_ASK_SPREAD: float = 0.0002
    EXECUTION_FUNDING_PER_TRADE: float = 0.0001

    # Optuna Sizing parameters
    OPTUNA_TUNING_ENABLED: bool = True
    OPTUNA_TRIALS: int = 15
    OPTUNA_SEED: int = 42
    OPTUNA_STORAGE_URL: str = ""
    OPTUNA_MIN_TRADES: int = 20
    # Deprecated aliases retained so existing deployments keep their calibrated costs.
    OPTUNA_COMMISSION: float | None = None
    OPTUNA_SLIPPAGE: float | None = None
    OPTUNA_FUNDING_PER_TRADE: float | None = None
    # Сколько последних (самых свежих) фолдов Walk-Forward использовать
    # при тюнинге. None = все фолды (старое поведение, дорого).
    # Ограничение снижает cost тюнинга и одновременно фокусирует подбор
    # параметров на актуальном рыночном режиме, а не на истории полугодовой давности.
    OPTUNA_MAX_FOLDS: int | None = 8
    # Nested WFO reserves these newest chronological segments. Optuna can only
    # access the older inner-train segment; risk calibration cannot access test.
    WFO_CALIBRATION_FRACTION: float = 0.20
    WFO_ECONOMIC_TEST_FRACTION: float = 0.20

    # Model Rollback SRE parameters
    ROLLBACK_CHECK_WINDOW: int = 10
    ROLLBACK_WIN_RATE_THRESHOLD: float = 0.35
    ROLLBACK_MAX_DRAWDOWN_THRESHOLD: float = 0.15

    ACTIVE_CONFIGS: list[tuple[str, str]] = [
        ("BTC/USDT", "1h"),
        ("ETH/USDT", "1h"),
        ("SOL/USDT", "1h"),
    ]

    CALIBRATION_MIN_TRADES: int = 10
    ECONOMIC_GATE_MIN_TRADES: int = 10

    # Trading Mode
    TRADING_MODE: Literal["paper", "shadow", "testnet", "mainnet"]

    # Binance API
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_PROXY: str = ""

    META_LABELING_ENABLED: bool = True
    META_LABELING_THRESHOLD: float = 0.5
    META_LABELING_MIN_TRADES: int = 30

    ATR_RISK_MODEL_ENABLED: bool = True
    LABEL_TP_ATR_MULT: float = 1.5
    LABEL_SL_ATR_MULT: float = 1.0

    def get_model_path(self, symbol: str, timeframe: str) -> str:
        """Динамически рассчитывает путь к pkl-файлу модели."""
        clean_symbol = symbol.replace("/", "").replace(":", "")
        clean_tf = timeframe.replace("/", "")
        return f"models/saved_models/lgbm_{clean_symbol}_{clean_tf}.pkl"

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

    @model_validator(mode="after")
    def validate_live_api_keys(self):
        if self.TRADING_MODE in {"testnet", "mainnet"}:
            if not self.BINANCE_API_KEY.strip() or not self.BINANCE_API_SECRET.strip():
                raise ValueError(
                    "Binance API Key и Secret обязательны для testnet/mainnet."
                )
        return self

    @property
    def SHADOW_TRADING(self) -> bool:
        return self.TRADING_MODE == "shadow"

    @property
    def LIVE_TRADING(self) -> bool:
        return self.TRADING_MODE in {"testnet", "mainnet"}

    @property
    def BINANCE_TESTNET(self) -> bool:
        return self.TRADING_MODE != "mainnet"

    @property
    def db_url(self) -> str:
        if self.DATABASE_URL:
            url = self.DATABASE_URL
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            return url
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
