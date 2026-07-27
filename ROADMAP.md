
---

# 🚀 Документ 3: Production Roadmap v1.0

Этот план предназначен для пошаговой реализации без архитектурного дрейфа.

---

### Фаза 1: Базовый каркас, БД и Биржа (1–2 день)
> **Цель:** Поднять структуру, подключить SQLite WAL и безопасный коннектор к Binance.

- [ ] **1.1. Конфиг и БД (`src/config.py`, `src/db.py`)**
  - Описать `Settings` на `pydantic-settings` (без Redis).
  - Настроить SQLAlchemy Async Engine для SQLite (`journal_mode=WAL`).
  - Создать минимальные модели: `Kline`, `Trade`, `PredictionLog`.
- [ ] **1.2. Биржевой Адаптер (`src/exchange/`)**
  - Описать `BaseExchange` (абстрактный класс).
  - Реализовать `BinanceExchange` (CCXT Futures):
    - Получение баланса, позиций, свечей.
    - Исполнение Market-ордеров.
    - Установка условных стоп-ордеров через **Algo Order API** (`STOP_MARKET`, `TAKE_PROFIT_MARKET` c `reduceOnly=True`).
  - Написать `PaperExchange` для локального тестирования.
- [ ] **1.3. Шина Событий (`src/event_bus.py`)**
  - Реализовать `AsyncEventBus` и `dataclasses` событий: `CandleClosedEvent`, `SignalEmittedEvent`, `OrderApprovedEvent`, `OrderExecutedEvent`, `TradeClosedEvent`, `ErrorEvent`.

---

### Фаза 2: Торговый Движок (Trading Core) (1–2 день)
> **Цель:** Собрать и связать изолированные сервисы в линейный пайплайн.

- [ ] **2.1. Модуль Стратегии (`src/strategy/`)**
  - `features.py`: Расчет индикаторов (`rsi`, `macd`, `atr`, `adx`, `bb`, `volatility`) с приведением к `float32`.
  - `model.py` (`Predictor`): Загрузка `active.pkl` и инференс с проверкой порога `MIN_EXPECTED_RETURN`.
  - `generator.py` (`SignalGenerator`): Принятие решения (свечи → фичи → модель → сигнал).
- [ ] **2.2. Модуль Рисков (`src/risk/`)**
  - `guards.py` (`RiskGuard`): Проверка Circuit Breaker (серия убытков), дневного лимита просадки и макс. позиций.
  - `sizer.py`: Расчет лота (% от свободного депозита) и цен SL/TP (ATR или %).
- [ ] **2.3. Подписчики Шины (`src/services/`)**
  - `MarketService`: Сбор свечей по таймеру → запись в БД → `CandleClosedEvent`.
  - `StrategyService`: `CandleClosed` → `SignalGenerator` → запись `PredictionLog` → `SignalEmittedEvent`.
  - `RiskService`: `SignalEmitted` → `RiskGuard` + `Sizer` → `OrderApprovedEvent`.
  - `ExecutionService`: `OrderApproved` → ордер на биржу + Algo Стопы → запись `Trade` → `OrderExecutedEvent`.
  - `ReconciliationService`: Фоновый процесс (раз в 60 сек) для сверки позиций на Binance с БД SQLite и закрытия «фантомных» сделок.

---

### Фаза 3: Telegram Бот и SRE Мониторинг (1 день)
> **Цель:** Управление, алерты и защита бота от падений.

- [ ] **3.1. Telegram Бот (`src/bot/`)**
  - Подключить `Aiogram 3` с `MemoryStorage`.
  - Реализовать команды: `/status` (баланс, режим, активная модель), `/positions` (открытые позиции), `/trades` (история), `/risk` (лимиты).
- [ ] **3.2. SRE & Notifier (`src/services/notifier_service.py`, `nexus_sdk/`)**
  - Отправка алертов об открытии/закрытии позиций в Telegram.
  - Подключение `NexusSDK`: Heartbeat каждые 15 сек, автоматический репорт незахваченных исключений.

---

### Фаза 4: Конвейер Самообучения (Retrain Lab) (1–2 дня)
> **Цель:** Изолированный скрипт обучения модели без риска забить OOM на 2 ГБ RAM.

- [ ] **4.1. Скрипт Обучения (`scripts/train.py`)**
  - Загрузка исторической истории свечей из SQLite в `float32`.
  - Формирование Post-Cost таргетов (учитывающих комиссию и SL/TP).
  - Обучение `EconomicReturnRegressor` (LightGBM) с параметром `OMP_NUM_THREADS=1`.
  - Walk-Forward валидация.
- [ ] **4.2. Economic Quality Gate**
  - Проверка кандидата на Holdout-фолде: Expectancy > 0, Profit Factor > 1.15.
- [ ] **4.3. Атомарное обновление (Hot-Reload)**
  - При успехе: атомарная перезапись `models/active.pkl`.
  - В `StrategyService`: перед инференсом проверка `os.path.getmtime("models/active.pkl")`. Если файл изменился — перечитка модели в память.
  - При неудаче: запись причины брака в лог, отправка отчета в Telegram.

---

### Фаза 5: Деплоймент и Запуск (1 день)
> **Цель:** Упаковка в Docker, CI/CD и вывод на реальный VPS.

- [ ] **5.1. Docker & Docker Compose (`Dockerfile`, `docker-compose.yml`)**
  - Минималистичный Dockerfile на `python:3.12-slim` с установленным `libgomp1` (для LightGBM).
  - Монтирование папок `./data` (для SQLite) и `./models`.
- [ ] **5.2. CI/CD (`.github/workflows/deploy.yml`)**
  - Автоматический прогон `pytest`.
  - Сборка и отправка образа в GHCR.
  - Деплой на VPS по SSH с установкой флага обслуживания в Nexus Redis (подавление ложных алертов при деплое).

---
