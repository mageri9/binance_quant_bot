from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import scripts.backfill as backfill_module


@pytest.mark.asyncio
async def test_backfill_paginates_and_advances_since(monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls, _timezone):
            return datetime(1970, 1, 1, 0, 0, 10, tzinfo=timezone.utc)

    class Exchange:
        def __init__(self):
            self.exchange = self
            self.calls: list[tuple[int, int]] = []
            self.closed = False

        async def fetch_ohlcv(self, _symbol, *, timeframe, since, limit):
            assert timeframe == "1h"
            self.calls.append((since, limit))
            pages = [
                [[1_000, 1, 2, 0, 1.5, 10]],
                [[2_000, 1.5, 3, 1, 2.5, 20]],
                [],
            ]
            return pages[len(self.calls) - 1]

        async def close(self):
            self.closed = True

    class Session:
        def __init__(self):
            self.executed = []
            self.commits = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, statement):
            self.executed.append(statement)

        async def commit(self):
            self.commits += 1

    exchange = Exchange()
    session = Session()

    async def init_db():
        return None

    monkeypatch.setattr(backfill_module, "datetime", FixedDateTime)
    monkeypatch.setattr(backfill_module, "init_db", init_db)
    monkeypatch.setattr(backfill_module, "get_settings", lambda: SimpleNamespace(BINANCE_API_KEY="", BINANCE_API_SECRET="", TRADING_MODE="mainnet"))
    monkeypatch.setattr(backfill_module, "BinanceExchange", lambda **_kwargs: exchange)
    monkeypatch.setattr(backfill_module, "AsyncSessionFactory", lambda: session)

    await backfill_module.backfill("BTC/USDT", "1h", days=1)

    assert [limit for _, limit in exchange.calls] == [1000, 1000, 1000]
    assert exchange.calls[1][0] == 1_001
    assert exchange.calls[2][0] == 2_001
    assert len(session.executed) == 2
    assert session.commits == 2
    assert exchange.closed
