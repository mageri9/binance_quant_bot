import os
import pytest

# 1. Принудительно изолируем тесты от реальной сети и локального .env
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["BOT_TOKEN"] = "123456:fake_token_for_tests"
os.environ["ADMIN_IDS"] = "[123456789]"
os.environ["TRADING_MODE"] = "paper"
os.environ["NEXUS_APP_SECRET"] = ""

from src.db import Base, engine


@pytest.fixture(autouse=True)
async def setup_test_db():
    """Автоматически инициализирует чистую БД в памяти перед каждым тестом."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)