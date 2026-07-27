import sqlite3

from src.db import configure_sqlite_connection


def test_configure_sqlite_connection_enables_wal_and_busy_timeout(tmp_path) -> None:
    database_path = tmp_path / "marketmind.db"
    connection = sqlite3.connect(database_path)
    try:
        configure_sqlite_connection(connection, None)

        assert connection.execute("PRAGMA journal_mode;").fetchone() == ("wal",)
        assert connection.execute("PRAGMA busy_timeout;").fetchone() == (5000,)
    finally:
        connection.close()
