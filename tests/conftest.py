import os
# Принудительная инициализация переменных окружения для прохождения тестов без ошибок валидации Pydantic
os.environ.setdefault("BOT_TOKEN", "123456:fake_token_for_ci_tests")
os.environ.setdefault("ADMIN_IDS", "[12345678]")
os.environ.setdefault("TRADING_MODE", "testnet")
os.environ.setdefault("BINANCE_API_KEY", "fake_api_key_for_tests")
os.environ.setdefault("BINANCE_API_SECRET", "fake_api_secret_for_tests")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from src.core.db import Base


@pytest_asyncio.fixture
async def temp_db_session():
    # Настраиваем изолированную временную БД в памяти
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()