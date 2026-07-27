
```markdown

# 📂 Структура Проекта MarketMind v1.0

Дерево каталогов и файлов проекта, соответствующее Стандарту Архитектуры v1.0 (2 ГБ RAM, Event-Driven Core + Изолированный Retrain Lab).

```text
MarketMind/
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD: pytest, сборка Docker и SSH-деплой на VPS
│
├── models/                       # Директория артефактов моделей (не коммитится)
│   ├── active.pkl                # Текущая Production-модель в инференсе
│   ├── candidate.pkl             # Обученный кандидат (до прохождения Quality Gate)
│   └── metadata.json             # Метаданные активной модели (версия, OOS метрики)
│
├── nexus_sdk/                    # SRE Мониторинг SDK
│   ├── __init__.py
│   └── error_handler.py          # Heartbeat, HMAC подпись, перехват исключений
│
├── scripts/                      # Изолированные CLI-скрипты (Retrain Lab)
│   ├── backfill.py               # Выкачивание исторической истории свечей с Binance
│   └── train.py                  # Изолированный Retrain Lab (LightGBM, Quality Gate, Hot-Reload)
│
├── src/                          # Торговый Движок (Trading Core — 24/7)
│   ├── bot/                      # Telegram-интерфейс (Aiogram 3, MemoryStorage)
│   │   ├── handlers.py           # Команды /status, /positions, /trades, /risk
│   │   └── keyboards.py          # Интерактивное меню
│   │
│   ├── exchange/                 # Адаптеры биржевого исполнения (Изоляция CCXT)
│   │   ├── base.py               # Абстрактный интерфейс BaseExchange
│   │   ├── binance.py            # Коннектор Binance Futures (Testnet/Mainnet + Algo Orders API)
│   │   └── paper.py              # Локальный эмулятор симуляции исполнения
│   │
│   ├── risk/                     # Модуль Управления Рисками
│   │   ├── guards.py             # RiskGuard (Circuit Breaker, Daily Loss, Max Positions)
│   │   └── sizer.py              # Position Sizer (Расчет размера лота и ATR/pct стопов)
│   │
│   ├── services/                 # Исполнители событий (Event Handlers)
│   │   ├── execution_service.py  # Исполнение ордеров на бирже + постановка Algo Стопов
│   │   ├── market_service.py     # Поллинг свечей -> Сохранение в SQLite -> CandleClosedEvent
│   │   ├── notifier_service.py   # Отправка алертов в Telegram и ошибок в Nexus SRE
│   │   ├── reconciliation_service.py # Сверка реальных позиций Binance с БД (раз в 60с)
│   │   ├── risk_service.py       # Валидация сигналов через RiskGuard + Sizer
│   │   └── strategy_service.py   # Генерация сигналов (Inference .pkl + PredictionLog)
│   │
│   ├── strategy/                 # Математика стратегии и ИИ Инференс
│   │   ├── features.py           # Расчет технических индикаторов (float32)
│   │   ├── generator.py          # SignalGenerator (Свечи -> Фичи -> Модель -> Сигнал)
│   │   └── model.py              # Predictor (Легкий инференс active.pkl)
│   │
│   ├── config.py                 # Настройки Pydantic Settings (без Redis)
│   ├── db.py                     # SQLAlchemy 2.0 Async + SQLite WAL (Kline, Trade, PredictionLog)
│   ├── event_bus.py              # Легковесный AsyncEventBus (in-memory)
│   └── main.py                   # Точка входа, инициализация шины и фоновые задачи
│
├── tests/                        # Набор Pytest тестов
│   ├── conftest.py               # Фикстура чистого SQLite в памяти (:memory:)
│   ├── test_event_bus.py         # Тесты шины событий
│   ├── test_exchange.py          # Тесты биржевых адаптеров
│   ├── test_pipeline.py          # Интеграционный тест торговой цепочки
│   ├── test_risk.py              # Тесты RiskGuard и Sizer
│   └── test_strategy.py          # Тесты индикаторов и генерации сигналов
│
├── .dockerignore
├── .env.example                  # Шаблон переменных окружения
├── .gitignore
├── ARCHITECTURE.md               # Общая документация
├── DEPENDENCY_RULES.md           # Матрица запрещенных связей компонентов
├── Dockerfile                    # Сборка python:3.12-slim с libgomp1
├── docker-compose.yml            # Запуск бота в Docker
├── PROJECT_TREE.md               # Данный файл структуры проекта
├── README.md
├── ROADMAP.md                    # Пошаговый план разработки (Фазы 1-5)
├── STANDART.md                   # Фиксированный Стандарт Архитектуры v1.0
└── requirements.txt              # Зависимости Python
```
