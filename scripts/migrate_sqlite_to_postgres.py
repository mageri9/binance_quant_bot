import argparse
import asyncio

from sqlalchemy import create_engine, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from src.core.config import get_settings
from src.core.db import Base
from src.db import models  # noqa: F401


async def migrate(sqlite_path: str, pg_url: str):
    sync_engine = create_engine(f"sqlite:///{sqlite_path}")
    async_engine = create_async_engine(pg_url)

    async with async_engine.begin() as pg_conn:
        for table in Base.metadata.sorted_tables:
            with sync_engine.connect() as sqlite_conn:
                rows = [dict(r._mapping) for r in sqlite_conn.execute(select(table))]

            if not rows:
                print(f"{table.name}: 0 rows, skip")
                continue

            await pg_conn.execute(table.insert(), rows)

            result = await pg_conn.execute(text(f"SELECT count(*) FROM {table.name}"))
            pg_count = result.scalar_one()
            if pg_count != len(rows):
                raise RuntimeError(f"{table.name}: expected {len(rows)}, got {pg_count}")

            if "id" in table.c:
                await pg_conn.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{table.name}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table.name}), 1))"
                ))

            print(f"{table.name}: migrated {len(rows)} rows")

    await async_engine.dispose()
    sync_engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-path", default="src/db/db.db")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.DATABASE_URL:
        raise SystemExit("DATABASE_URL не задан — некуда мигрировать")

    asyncio.run(migrate(args.sqlite_path, settings.DATABASE_URL))