
---

# 🏛 Документ 1: Архитектурный Стандарт MarketMind v1.0

### 1. Архитектурный каркас
Система делится на два изолированных мира:
* **Trading Core (24/7):** Легковесный асинхронный процесс Python 3.12 (~100 МБ RAM).
* **Retrain Lab (Cron/On-Demand):** Изолированный короткоживущий подпроцесс (`OMP_NUM_THREADS=1`, `nice -n 19`, лимит RAM 1.2 ГБ). После завершения отдаёт 100% ресурсов.

```text
[ Binance Futures API ]
         │ (HTTP / Algo Orders)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ TRADING CORE (Single Async Process, ~100MB RAM)                 │
│                                                                 │
│  MarketService ──► StrategyService ──► RiskService ──► Execution │
│       │                 │                   │             │     │
│       ▼                 ▼                   ▼             ▼     │
│  [SQLite: Klines] [PredictionLog]      [RiskGuard]   [Trade/Ledger]
└─────────────────────────────────────────────────────────────────┘
```

### 2. Стек и хранение
* **Язык/Фреймворк:** Python 3.12+, Asyncio, Aiogram 3 (`MemoryStorage`).
* **База Данных:** SQLite в режиме `PRAGMA journal_mode=WAL;` + `PRAGMA busy_timeout=5000;`. (Нулевое потребление RAM на Сервер БД).
* **Шина Событий:** Простой `AsyncEventBus` (`dict[Type[Event], list[Callable]]`) без Outbox и без повторов.
* **ML:** LightGBM (`float32`), Scikit-learn, Pandas.

---
