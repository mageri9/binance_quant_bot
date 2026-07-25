import pytest
from src.core.config import Settings


def test_db_url_postgresql_conversion():
    # Проверка автоматического добавления asyncpg-драйвера к Postgres URL
    settings_pg = Settings(
        BOT_TOKEN="test_token",
        ADMIN_IDS=[123],
        DATABASE_URL="postgresql://user:pass@localhost:5432/dbname",
        TRADING_MODE="testnet",
        BINANCE_API_KEY="fake_key",
        BINANCE_API_SECRET="fake_secret"
    )
    assert settings_pg.db_url == "postgresql+asyncpg://user:pass@localhost:5432/dbname"


def test_db_url_old_postgres_conversion():
    # Проверка конвертации устаревшего префикса postgres://
    settings_old = Settings(
        BOT_TOKEN="test_token",
        ADMIN_IDS=[123],
        DATABASE_URL="postgres://user:pass@localhost:5432/dbname",
        TRADING_MODE="testnet",
        BINANCE_API_KEY="fake_key",
        BINANCE_API_SECRET="fake_secret"
    )
    assert settings_old.db_url == "postgresql+asyncpg://user:pass@localhost:5432/dbname"


def test_db_url_sqlite_default():
    # Проверка возврата SQLite по умолчанию при пустом DATABASE_URL
    settings_default = Settings(
        BOT_TOKEN="test_token",
        ADMIN_IDS=[123],
        DATABASE_URL="",
        TRADING_MODE="testnet",
        BINANCE_API_KEY="fake_key",
        BINANCE_API_SECRET="fake_secret"
    )
    assert settings_default.db_url.startswith("sqlite+aiosqlite:///")


def test_blank_backtest_risk_values_disable_dynamic_sizing():
    settings = Settings(
        BOT_TOKEN="test_token",
        ADMIN_IDS=[123],
        TRADING_MODE="paper",
        BACKTEST_STOP_RISK_PCT="",
        BACKTEST_TARGET_VOLATILITY="   ",
    )

    assert settings.BACKTEST_STOP_RISK_PCT is None
    assert settings.BACKTEST_TARGET_VOLATILITY is None
