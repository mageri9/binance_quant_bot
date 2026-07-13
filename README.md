# 🤖 MarketMind — Binance Quant Trading Bot

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Aiogram](https://img.shields.io/badge/Aiogram-3.x-green.svg)](https://docs.aiogram.dev/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5-orange.svg)](https://lightgbm.readthedocs.io/)

Telegram-бот для количественной торговли на Binance. Собирает исторические свечи,
считает технические индикаторы, обучает ML-модель (LightGBM) на честной
walk-forward валидации, торгует в режиме paper trading и самостоятельно
переобучается по расписанию — с защитой от деградации модели.

## ✨ Возможности

### Данные
- Загрузка OHLCV-свечей с Binance через `ccxt` с upsert-сохранением в SQLite
- Скрипт разового бэкфилла истории (`scripts/backfill.py`)
- Автоматический докач свежих свечей раз в час в фоне

### Признаки и разметка
- Индикаторы: RSI (сглаживание Уайлдера), MACD (+ сигнальная линия, гистограмма),
  волатильность, volume ratio
- Метки: бинарная (рост / не рост) и тройная (long / short / hold) с настраиваемым
  горизонтом и порогом доходности
- Датасеты версионируются и сохраняются в Parquet + JSON-паспорт (git sha, диапазон
  дат, список фич, размер выборки)

### ML-пайплайн
- **Walk-Forward Validation** — строго хронологическое разделение train/test, без
  утечек данных (покрыто отдельным leakage-тестом)
- Две модели на каждый прогон: **baseline** (LogisticRegression) и продовая
  (**LightGBM**)
- Логирование каждого эксперимента (параметры, метрики) в БД
- Инференс на последней свече (`Predictor`)

### Автообучение
- Фоновый цикл (`retrain_loop`, интервал настраивается): собирает свежий датасет →
  обучает baseline и LGBM → **продвигает новую модель в прод только если её F1
  превосходит baseline того же прогона**. Если нет — модель отбраковывается, а
  админам летит предупреждение (защита от скрытой деградации из-за data leakage
  или смены рыночного режима)

### Торговля (Paper Trading)
- Виртуальный портфель со стартовым балансом $10 000
- Открытие/закрытие сделок по сигналу модели с учётом Stop-Loss, Take-Profit и
  выхода по таймауту (horizon)
- Отдельный бэктест-симулятор стратегии (`simulate_strategy`) с расчётом комиссий
- Метрики стратегии: Win Rate, Profit Factor, Sharpe, Sortino, Max Drawdown,
  Expectancy, суммарная доходность

### Telegram-бот
| Команда | Описание |
|---|---|
| `/start` | Регистрация пользователя, главное меню |
| `/status` | Статус виртуального портфеля и открытой позиции |
| `/signals` | Ручной опрос ML-модели по текущим данным |
| `/report` | Отчёт по фактическим результатам paper trading |
| `/admin` | Админ-панель |
| `/stats` | Количество активных пользователей (только админ) |

### Инфраструктура
- **Redis**: FSM-хранилище, кэш пользователей, sliding-window rate limiting
  (Lua-скрипт), освобождение админов от лимита
- **Alembic**: версионируемые миграции БД
- **Nexus SRE SDK** (опционально): автоматическая отправка ошибок и heartbeat во
  внешнюю систему мониторинга по HMAC-подписанным вебхукам
- **CI/CD**: GitHub Actions — тесты → сборка Docker-образа в GHCR → деплой на VPS
  по SSH с установкой флага техобслуживания в Redis (подавление ложных алертов на
  время рестарта)
- Полное покрытие тестами: коллектор данных, фичи, лейблы, бэктест-сплиттер,
  baseline/LGBM обучение, paper trading, стратегия, хендлеры

## 🗂️ Структура проекта

```
src/
  core/
    config.py          # Pydantic Settings
    db.py               # SQLAlchemy async engine + session factory
    redis.py             # Redis client singleton
    router_manager.py    # Сборка роутеров aiogram
  crud/                 # Repository-паттерн: user, kline, experiment, paper
  data/
    collector.py        # Загрузка свечей с Binance (ccxt)
  datasets/
    build.py             # Сборка датасета: фичи + метки → Parquet/JSON
  db/
    models.py            # SQLAlchemy-модели
    migrations/          # Alembic
  features/
    engineering.py        # RSI, MACD, волатильность, volume ratio
  labels/
    generator.py          # Бинарные и тройные метки
  models/
    backtest.py           # TimeSeriesWalkForwardSplitter
    baseline.py            # LogisticRegression эксперимент
    train.py               # LightGBM эксперимент + сохранение модели
    predictor.py            # Инференс на новых свечах
  strategy/
    signals.py              # Метрики стратегии + симулятор бэктеста
  paper_trading/
    engine.py                # Движок виртуальной торговли
  handlers/               # admin/, user/ — message.py, callback.py
  keyboards/
  filters/
  middlewares/            # db, redis, rate_limit, logger
  main.py                 # Точка входа, фоновые циклы
nexus_sdk/                # Клиент для внешнего SRE-мониторинга
scripts/
  backfill.py              # Разовое наполнение БД историей
  migrate.sh               # Обёртка над alembic для Docker
tests/
```

## 🚀 Быстрый старт

### Локально

```bash
git clone <repo_url>
cd binance_quant_bot

cp .env.example .env
# Заполни BOT_TOKEN и ADMIN_IDS в .env

pip install -r requirements.txt

# Применить миграции
alembic upgrade head

# Наполнить БД историческими свечами (иначе на автообучение уйдут дни)
python -m scripts.backfill --symbol BTC/USDT --timeframe 1h --days 90

# Запуск (Redis должен быть доступен)
python -m src.main
```

### Docker

```bash
cp .env.example .env
# Заполни .env

docker compose up --build -d
```

## 🗄️ Миграции (Alembic)

```bash
# Новая миграция
alembic revision --autogenerate -m "add field"

# Применить
alembic upgrade head

# Откатить на шаг
alembic downgrade -1

# То же самое в Docker
./scripts/migrate.sh upgrade head
./scripts/migrate.sh revision --autogenerate -m "add field"
```

## 🔧 Конфигурация (.env)

```env
BOT_TOKEN=your_bot_token_here
ADMIN_IDS=[123456789,987654321]

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Rate limiting
RATE_LIMIT_CALLS=5
RATE_LIMIT_PERIOD=10

# Logging
LOG_LEVEL=INFO

# Nexus SRE (опционально)
NEXUS_APP_SECRET=your_nexus_app_secret_here
NEXUS_ENDPOINT_URL=http://nexus-webhook:8000/events/app
NEXUS_PROJECT_NAME=binance_quant_bot
```

Дополнительные параметры модели/переобучения задаются в `src/core/config.py`
(`MODEL_PATH`, `RETRAIN_INTERVAL_SECONDS`, `MIN_KLINES_FOR_TRAIN`, `TRAIN_SIZE`,
`TEST_SIZE`, `LABEL_HORIZON`, `LABEL_THRESHOLD`).

## 💉 Dependency Injection

Сессия БД и Redis прокидываются в хендлеры через middleware:

```python
async def my_handler(message: Message, session: AsyncSession, redis: Redis):
    ...
```

## ⚡ Стек

| Компонент | Технология |
|---|---|
| Фреймворк бота | Aiogram 3 |
| Биржевые данные | ccxt (Binance) |
| БД | SQLite + SQLAlchemy (async) |
| Миграции | Alembic |
| Кэш / FSM storage / Rate limit | Redis 7 |
| ML | scikit-learn, LightGBM |
| Данные | pandas, pyarrow (Parquet) |
| Конфиг | Pydantic Settings |
| Логи | Loguru |
| Мониторинг | Nexus SRE SDK (опционально) |
| Тесты | pytest, pytest-asyncio |
| Контейнеры | Docker + docker-compose |
| CI/CD | GitHub Actions → GHCR → деплой по SSH |

## ⚠️ Известные ограничения / TODO

- Торгуется только пара `BTC/USDT` на таймфрейме `1h`, без диверсификации
- Уровни Stop-Loss/Take-Profit в `paper_trading/engine.py` заданы константами
  «на глаз» и не провалидированы через `strategy.signals.simulate_strategy`
- Тройные метки (`target_triple`) генерируются, но пока не используются в обучении
- Реальные ордера на биржу не отправляются — только paper trading
- Команда `/broadcast` заявлена в архитектуре, но не реализована