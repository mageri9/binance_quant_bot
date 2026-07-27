# 🧊 MarketMind [ЗАМОРОЖЕН / FROZEN]

> **Статус проекта:** ❄️ **Архивирован / Заморожен**
> 
> Разработка и поддержка проекта приостановлены. Код, архитектурные стандарты и наработки исследовательского конвейера сохранены в репозитории как исторический архив.

---

## 📌 О проекте

**MarketMind** — исследовательская событийно-ориентированная платформа для алгоритмической торговли и машинного обучения на базе Python 3.12, LightGBM и Binance Futures.

Проект спроектирован с чётким разделением на два контура:
1. **Trading Core (24/7):** Легковесный асинхронный торговый движок на базе `AsyncEventBus`, `SQLite WAL` и `Aiogram 3`.
2. **Research Lab:** Модульный конвейер исследований и экспериментов с защитой от утечек данных (Purged Walk-Forward Cross-Validation).

---

## 🏗 Архитектура системы

```text
[ Binance Futures API ]
          │ (HTTP / Algo Orders)
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ TRADING CORE (Single Async Process, 24/7)                       │
│                                                                 │
│  MarketService ──► StrategyService ──► RiskService ──► Execution │
│       │                 │                   │             │     │
│       ▼                 ▼                   ▼             ▼     │
│  [SQLite: Klines] [PredictionLog]      [RiskGuard]   [Trade/Ledger]
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RESEARCH LAB (On-Demand / Experiment-Centric)                   │
│                                                                 │
│  dataset ──► labeling ──► validation ──► train ──► backtest     │
│     │                                                 │         │
│     ▼                                                 ▼         │
│  src/strategy/features.py ──────────────► artifacts/exp_XXXX/   │
└─────────────────────────────────────────────────────────────────┘