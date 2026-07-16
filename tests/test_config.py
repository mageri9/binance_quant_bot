import pytest
from src.core.config import Settings


def test_db_url_postgresql_conversion():
    # Проверка автоматического добавления asyncpg-драйвера к Postgres URL
    settings_pg = Settings(
        BOT_TOKEN="test_token",
        ADMIN_IDS=[123],
        DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
    )
    assert settings_pg.db_url == "postgresql+asyncpg://user:pass@localhost:5432/dbname"


def test_db_url_old_postgres_conversion():
    # Проверка конвертации устаревшего префикса postgres://
    settings_old = Settings(
        BOT_TOKEN="test_token",
        ADMIN_IDS=[123],
        DATABASE_URL="postgres://user:pass@localhost:5432/dbname"
    )
    assert settings_old.db_url == "postgresql+asyncpg://user:pass@localhost:5432/dbname"


def test_db_url_sqlite_default():
    # Проверка возврата SQLite по умолчанию при пустом DATABASE_URL
    settings_default = Settings(
        BOT_TOKEN="test_token",
        ADMIN_IDS=[123],
        DATABASE_URL=""
    )
    assert settings_default.db_url.startswith("sqlite+aiosqlite:///")