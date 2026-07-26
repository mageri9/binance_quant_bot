```markdown
# 🏛 Архитектурная документация MarketMind (Event-Driven Core)

MarketMind — это событийно-ориентированная платформа для алгоритмической торговли криптовалютными фьючерсами на базе машинного обучения (LightGBM) с поддержкой симуляции (`paper`), `testnet` и `mainnet` биржи Binance Futures.

---

## 🎯 Ключевые архитектурные принципы

1. **Легковесная In-Memory Шина Событий (AsyncEventBus):**
   * Все компоненты системы полностью изолированы (Decoupled) и общаются исключительно через строго типизированные `dataclass`-события.
   * Отсутствие тяжелых внешних брокеров (Kafka/RabbitMQ) и таблицы Transactional Outbox. Все события обрабатываются асинхронно в памяти одного процесса Python.

2. **Платформа + Стратегия как Плагин:**
   * Торговый движок (Исполнение, Риски, Телеграм, БД) ничего не знает о внутреннем устройстве стратегии.
   * Модуль `strategy/` является подключаемым плагином. Логика генерации сигналов изолирована и может легко заменяться (от простых правил до LightGBM или нейросетей).

3. **Принцип единой ответственности (SRP) и микро-модули:**
   * Отказ от громоздких монолитных файлов (нет файлов > 150-200 строк).
   * Чёткое разделение ответственности облегчает понимание кода разработчиком и устраняет галлюцинации LLM при рефакторинге.

4. **Мгновенное переключение режимов торговли (`TRADING_MODE`):**
   * Переключение между `paper` (локальная эмуляция), `testnet` и `mainnet` выполняется через единую переменную окружения `.env`.
   * Изменение режима подменяет только физический адаптер биржи (`PaperExchange` vs `BinanceExchange`), оставляя весь пайплайн обработки событий нетронутым.

5. **Встроенный SRE-Мониторинг (Nexus SDK):**
   * Прямая интеграция с Nexus SRE SDK для периодической отправки Heartbeat и автоматического репортинга критических ошибок с контекстом выполнения.

---

## 📂 Структура проекта

```text
MarketMind/
├── nexus_sdk/                # Ваш SRE SDK (Heartbeat, HMAC, Error Reporting)
│   ├── __init__.py
│   └── error_handler.py
│
├── src/
│   ├── config.py             # Настройки Pydantic (токены, риски, пары, режимы)
│   ├── event_bus.py          # In-Memory AsyncEventBus + Dataclasses событий
│   ├── db.py                 # SQLAlchemy 2.0 Async (Kline, Trade, PredictionLog)
│   │
│   ├── strategy/             # Плагин Стратегии
│   │   ├── features.py       # Расчет индикаторов (RSI, ATR, MACD, ADX, Lagged Returns)
│   │   ├── model.py          # LightGBM EconomicReturnRegressor & Predictor
│   │   └── generator.py      # Принятие решений (Candles -> Features -> Model -> Signal)
│   │
│   ├── risk/                 # Модуль Риск-Менеджмента
│   │   ├── sizer.py          # Расчет размера позиции (% от банка, ATR стопы)
│   │   └── guards.py         # Circuit Breaker, дневная просадка, лимит позиций
│   │
│   ├── exchange/             # Адаптеры Рыночных Площадок
│   │   ├── base.py           # Абстрактный интерфейс BaseExchange
│   │   ├── paper.py          # Локальный эмулятор исполнения
│   │   └── binance.py        # Binance Futures CCXT (Testnet/Mainnet) + Algo Order API
│   │
│   ├── services/             # Подписчики события (Event Pipeline)
│   │   ├── market.py         # Сборщик свечей -> CandleClosedEvent
│   │   ├── strategy_service.py# CandleClosedEvent -> SignalEmittedEvent
│   │   ├── risk_service.py   # SignalEmittedEvent -> OrderApprovedEvent
│   │   ├── execution_service.py# OrderApprovedEvent -> OrderExecutedEvent
│   │   └── notifier_service.py # Telegram уведомления + Nexus SRE
│   │
│   ├── bot/                  # Telegram Бот (Aiogram 3.x)
│   │   ├── handlers.py       # Команды /status, /positions, /trades, /risk
│   │   └── keyboards.py      # Интерактивные кнопки
│   │
│   └── main.py               # Точка входа, инициализация шины и фоновые процессы
│
├── scripts/                  # Вспомогательные скрипты
│   ├── backfill.py           # Загрузка исторической истории свечей с Binance
│   └── train.py              # Обучение модели LightGBM Economic Return
│
└── tests/                    # Набор модульных и интеграционных тестов (Pytest)
    ├── conftest.py           # Фикстура изоляции окружения в памяти
    ├── test_event_bus.py     # Тесты шины событий
    ├── test_strategy_and_risk.py # Тесты индикаторов и рисков
    └── test_pipeline_integration.py # Интеграционный тест торговой цепочки
```

---

## 🔄 Пайплайн обработки событий (Event Data Flow)

Все торговое поведение описывается линеаризованной цепочкой событий:

```text
[ Binance / Exchange ]
          │
          ▼ (HTTP / Polling)
┌──────────────────┐
│  MarketService   │ ──► Публикует: CandleClosedEvent
└──────────────────┘
          │
          ▼
┌──────────────────┐
│ StrategyService  │ ──► Публикует: SignalEmittedEvent (при signal != 0)
└──────────────────┘     Записывает: PredictionLog в БД
          │
          ▼
┌──────────────────┐
│   RiskService    │ ──► Валидирует: RiskGuard (Circuit Breaker / Daily Loss)
└──────────────────┘     Рассчитывает: Размер лота и SL/TP
          │              Публикует: OrderApprovedEvent
          ▼
┌──────────────────┐
│ ExecutionService │ ──► Исполняет ордер через BaseExchange (Paper / Binance Algo API)
└──────────────────┘     Записывает: Trade (OPEN) в БД
          │              Публикует: OrderExecutedEvent / TradeClosedEvent
          ▼
┌──────────────────┐
│ NotifierService  │ ──► Отправляет форматированное сообщение в Telegram
└──────────────────┘     При ошибках репортит в Nexus SRE SDK
```

---

## 🗄 Схема базы данных (`src/db.py`)

База данных содержит всего **3 прозрачные таблицы**:

1. **`klines`**:
   * История OHLCV-свечей (`symbol`, `timeframe`, `open_time`, `open`, `high`, `low`, `close`, `volume`).
2. **`trades`**:
   * Журнал открытых и закрытых позиций (`symbol`, `status`, `side`, `entry_price`, `exit_price`, `amount`, `sl_price`, `tp_price`, `pnl`, `mode`, `order_id`).
3. **`prediction_logs`**:
   * История всех выданных прогнозов модели для MLOps-аналитики (`symbol`, `timeframe`, `signal`, `expected_return`, `price`).

---

## 🛠 Руководство разработчика

### Запуск тестов:
```bash
python -m pytest tests/
```

### Наполнение базы историческими данными:
```bash
python -m scripts.backfill --symbol BTC/USDT --timeframe 1h --days 30
```

### Обучение модели LightGBM:
```bash
python -m scripts.train --symbol BTC/USDT --timeframe 1h
```

### Запуск торгового бота:
```bash
python -m src.main
```