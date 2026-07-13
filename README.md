# 🤖 Aiogram 3 — Advanced Template

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Aiogram](https://img.shields.io/badge/Aiogram-3.x-green.svg)](https://docs.aiogram.dev/)

Продвинутый шаблон для Telegram-ботов с Redis, SQLAlchemy, FSM и Docker.

## 🗂️ Структура

```
src/
  core/
    config.py          # Pydantic Settings (lru_cache singleton)
    db.py              # SQLAlchemy async engine + session factory
    redis.py           # Redis client singleton
    router_manager.py  # Сборка роутеров
  crud/
    user.py            # UserRepository — паттерн Repository
  db/
    models.py          # SQLAlchemy модели
    migrations/        # Alembic миграции
  filters/
    chat_type.py
    check_admin.py
  handlers/
    admin/             # callback.py, message.py
    user/              # callback.py, message.py (FSM пример)
  keyboards/
    user.py            # InlineKeyboardBuilder / ReplyKeyboardBuilder
  middlewares/
    db.py              # Inject AsyncSession в хендлеры
    redis.py           # Inject Redis в хендлеры
    rate_limit.py      # Sliding window rate limiter (Redis)
    logger.py          # Структурированный лог каждого апдейта
  services/
    user.py            # UserService — бизнес-логика + кэш Redis
  states/
    user.py            # FSM States (пример: FeedbackForm)
  main.py
```

## 🚀 Быстрый старт

### Локально

```bash
git clone ...
cd bot

cp .env.example .env
# Заполни BOT_TOKEN и ADMIN_IDS в .env

pip install -r requirements.txt

# Применить миграции (создаёт таблицы)
alembic upgrade head

# Запустить (Redis должен быть доступен)
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
# Создать новую миграцию
alembic revision --autogenerate -m "add users table"

# Применить
alembic upgrade head

# Откатить на шаг
alembic downgrade -1

# В Docker через скрипт
./scripts/migrate.sh upgrade head
./scripts/migrate.sh revision --autogenerate -m "add field"
```

## ⚡ Стек

| Компонент | Технология |
|---|---|
| Фреймворк | Aiogram 3 |
| БД | SQLite + SQLAlchemy async |
| Миграции | Alembic |
| Кэш / FSM storage | Redis 7 |
| Конфиг | Pydantic Settings |
| Логи | Loguru |
| Контейнеры | Docker + docker-compose |

## 🔐 Rate Limiting

Sliding window через Redis Lua-скрипт. Настраивается в `.env`:
```env
RATE_LIMIT_CALLS=5    # кол-во запросов
RATE_LIMIT_PERIOD=10  # за N секунд
```
Админы из `ADMIN_IDS` освобождены от лимита.

## 💉 Dependency Injection

Сессия БД и Redis прокидываются в хендлеры через middleware — просто объяви параметры:

```python
async def my_handler(message: Message, session: AsyncSession, redis: Redis):
    ...
```

## 🔧 .env

```env
BOT_TOKEN=...
ADMIN_IDS=[123456789]

REDIS_HOST=redis        # redis — в Docker, localhost — локально
REDIS_PORT=6379
REDIS_PASSWORD=

RATE_LIMIT_CALLS=5
RATE_LIMIT_PERIOD=10

LOG_LEVEL=INFO
```
