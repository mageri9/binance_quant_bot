# MarketMind

Асинхронный торговый бот для Binance Futures (Python 3.12, aiogram 3, LightGBM).

## Статус

Прототип. Trading Core реализован и покрыт тестами. Модель торгового сигнала
**ещё не доказала положительное мат. ожидание** — это следующий и единственный
приоритет перед реальными деньгами. См. `ROADMAP.md`.

## Архитектура

Один процесс, событийная шина, три режима биржи: `paper` (симуляция в SQLite),
`testnet`, `mainnet`. Подробности — `ARCHITECTURE.md`.

## Быстрый старт

```bash
cp .env.example .env        # заполнить BOT_TOKEN, ADMIN_IDS
python -m pytest tests/     # тесты должны быть зелёными
python -m scripts.backfill --symbol BTC/USDT --timeframe 1h --days 90
python -m scripts.train --symbol BTC/USDT --timeframe 1h
python -m src.main          # TRADING_MODE=paper в .env
```

## Правило перед mainnet

Модель включается в боевой режим только если прошла gate в `scripts/train.py`
(п. 1 в `ROADMAP.md`) **и** отработала в `paper`-режиме минимум 1–2 недели
с положительным суммарным PnL по таблице `trades` на 20+ сделках.
Без этого — TRADING_MODE остаётся `paper` или `testnet`.