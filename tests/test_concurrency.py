import asyncio
import os
import tempfile
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import event

from src.core.db import Base
from src.crud.kline import KlineRepository
from src.crud.paper import PaperTradingRepository
from src.crud.experiment import ExperimentRepository


@pytest.mark.asyncio
async def test_concurrent_multi_session_db_stress():
    """
    Стресс-тест на параллельную работу нескольких независимых сессий БД.
    Проверяет отсутствие взаимных блокировок и корректность работы AsyncRLock
    при конкурентных изменениях баланса портфеля.

    Использует временный файл на диске с включенным режимом WAL для полноценной
    изоляции транзакций на уровне параллельных сессий (как в PostgreSQL).
    """
    # Создаем временный файл базы данных
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)  # Закрываем дескриптор, чтобы им мог управлять SQLAlchemy

    db_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(db_url, echo=False)

    # Настраиваем WAL и busy_timeout, аналогично src/core/db.py
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Инициализируем портфель один раз перед запуском конкурентных задач
        async with session_factory() as session:
            repo = PaperTradingRepository(session)
            await repo.get_portfolio()

        # Воркер для записи свечей
        async def kline_worker(worker_id: int):
            for i in range(10):
                async with session_factory() as session:
                    repo = KlineRepository(session)
                    klines_data = [
                        {
                            "symbol": f"SYM_{worker_id}",
                            "timeframe": "1h",
                            "open_time": 1000 + i * 3600 * 1000,
                            "open": 100.0 + i,
                            "high": 105.0 + i,
                            "low": 95.0 + i,
                            "close": 101.0 + i,
                            "volume": 500.0,
                        }
                    ]
                    await repo.save_klines(klines_data)
                await asyncio.sleep(0.005)

        # Воркер для торговли (конкурентное открытие/закрытие сделок)
        async def trade_worker(worker_id: int):
            symbol = f"TRADE_{worker_id}"
            for i in range(5):
                # Открытие сделки
                async with session_factory() as session:
                    repo = PaperTradingRepository(session)
                    await repo.create_trade(
                        symbol=symbol,
                        entry_price=10.0 * (worker_id + 1),
                        amount=1.0,
                        sl_price=9.0 * (worker_id + 1),
                        tp_price=12.0 * (worker_id + 1),
                        entry_candle_time=2000 + i,
                    )
                await asyncio.sleep(0.005)

                # Закрытие сделки с прибылью
                async with session_factory() as session:
                    repo = PaperTradingRepository(session)
                    active = await repo.get_active_trade(symbol)
                    if active:
                        await repo.close_trade(
                            active,
                            exit_price=11.0 * (worker_id + 1),
                            pnl=1.0 * (worker_id + 1),
                        )
                await asyncio.sleep(0.005)

        # Воркер для логирования экспериментов (имитация параллельного ретрейна)
        async def experiment_worker(worker_id: int):
            for i in range(5):
                async with session_factory() as session:
                    repo = ExperimentRepository(session)
                    await repo.log_experiment(
                        model_name=f"Model_{worker_id}",
                        dataset_version=f"v{i}",
                        parameters={"p": i},
                        metrics={"f1": 0.5 + (i * 0.05)},
                        git_sha="abcdef123456",
                    )
                await asyncio.sleep(0.005)

        # Собираем пачку конкурентных задач (5 воркеров каждого типа)
        tasks = []
        for i in range(5):
            tasks.append(kline_worker(i))
            tasks.append(trade_worker(i))
            tasks.append(experiment_worker(i))

        # Запускаем все задачи параллельно
        await asyncio.gather(*tasks)

        # Проверяем консистентность итоговых данных
        async with session_factory() as session:
            pt_repo = PaperTradingRepository(session)
            portfolio = await pt_repo.get_portfolio()

            # Суммарный PnL: 5 * (1 + 2 + 3 + 4 + 5) = 75.0$
            # Итоговый баланс: 10000 + 75 = 10075.0$
            assert portfolio.balance == 10075.0
            assert portfolio.positions_value == 0.0

            # Проверяем, что записались ровно 25 закрытых сделок (5 воркеров * 5 сделок)
            all_closed = []
            for i in range(5):
                closed = await pt_repo.get_closed_trades(f"TRADE_{i}")
                all_closed.extend(closed)
            assert len(all_closed) == 25

    finally:
        # Корректно закрываем движок и удаляем временные файлы SQLite (включая WAL-логи)
        await engine.dispose()
        for suffix in ["", "-wal", "-shm"]:
            fpath = db_path + suffix
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass