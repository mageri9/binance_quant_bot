```markdown
# 🤖 MarketMind — Binance Quant Trading Bot

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Aiogram](https://img.shields.io/badge/Aiogram-3.x-green.svg)](https://docs.aiogram.dev/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5-orange.svg)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-4.0-blue.svg)](https://optuna.org/)

Telegram-бот для автоматизированной количественной торговли на Binance в режиме виртуального портфеля (Paper Trading). Бот параллельно обрабатывает три актива (`BTC/USDT`, `ETH/USDT`, `SOL/USDT`), обучает многоклассовую модель машинного обучения (LightGBM) на честной walk-forward валидации с подбором гиперпараметров через Optuna, торгует в обе стороны (Long/Short) и защищен от деградации показателей с помощью SRE-системы автоматического отката на резервные бэкапы.

---

## ✨ Возможности

### 📈 Мультиактивная торговля в обе стороны (LONG / SHORT)
* Параллельное ведение торговли по трем ликвидным криптоактивам: **`BTC/USDT`**, **`ETH/USDT`** и **`SOL/USDT`** на таймфрейме `1h`.
* Открытие и закрытие сделок по сигналам модели с расчетом Stop-Loss, Take-Profit и выходом по временному тайм-ауту (horizon).
* Математическое вычисление направления позиции (LONG/SHORT) без расширения схемы базы данных SQLite для 100% обратной совместимости.
* Сводный интерактивный статус портфеля и агрегированная статистика по всем закрытым сделкам (Sharpe, Sortino, Win Rate, Drawdown, Profit Factor).

### ⚙️ Управление капиталом и рисками (Money Management)
* **Динамический сайзинг:** Объем каждой сделки рассчитывается как заданный процент от текущего общего баланса портфеля (например, ровно 10% от капитала).
* Защита от микро-ордеров с помощью настраиваемого минимального объема сделки в долларах.
* Защита базы данных от ухода баланса портфеля в минус.

### 🔬 Признаки, разметка и подбор параметров
* **Расширенные признаки:** RSI (Wilder), MACD, историческая волатильность, Volume Ratio, Bollinger Bands, ATR и ADX, написанные с нуля на Pandas и NumPy без внешних зависимостей.
* **Тройная разметка (target_triple):** Обучение модели предсказывать 3 рыночных состояния: падение (`-1.0` -> класс `0`), флэт (`0.0` -> класс `1`) и рост (`1.0` -> класс `2`).
* **Оптимизация параметров:** Встроенный автоматический тюнинг гиперпараметров LightGBM через **Optuna** по метрике F1-macro на кросс-валидации.
* Полное покрытие тестами на утечку данных в будущее (Data Leakage Protection).

### 🚨 SRE-автооткат при деградации результатов
* Автоматическое создание бэкапов стабильных файлов моделей перед заменой при переобучении.
* Фоновая проверка результатов торговли: если Win Rate за последнее окно сделок падает ниже порога или просадка превышает норму, бот автоматически откатывает рабочую модель на последний успешный бэкап и отправляет критическое оповещение администраторам.
* Блокировка повторных проверок (кулдаун) в Redis на 24 часа после совершения отката для накопления свежей статистики.

### 💬 Администрирование и Telegram UI
* Полностью кнопочный интерактивный интерфейс. Динамические кнопки подписки и отписки меняют состояние в реальном времени.
* Защищенный FSM-режим рассылки сообщений `/broadcast` всем активным пользователям бота с автоматическим отслеживанием блокировок и изменением статуса активности пользователя в SQLite.

---

## 🗂️ Структура проекта

```
scripts/
  backfill.py          # Разовое наполнение БД историческими свечами
  calibrate.py         # Сеточный бэктест-поиск оптимальных параметров SL/TP
  migrate.sh           # Запуск миграций Alembic в Docker
src/
  core/
    config.py          # Настройки Pydantic Settings
    db.py              # async сессии SQLAlchemy + SQLite
    redis.py           # Инициализация клиента Redis
    router_manager.py  # Сборка роутеров
  crud/
    user.py, kline.py, experiment.py, paper.py  # Repository-паттерны
  data/
    collector.py        # Загрузка свежих свечей через ccxt
  datasets/
    build.py           # Сборка версионированных датасетов в Parquet
  db/
    models.py          # SQLAlchemy-модели таблиц БД
    migrations/        # Alembic миграции
  features/
    engineering.py     # RSI, MACD, BB, ATR, ADX (чистый Pandas/NumPy)
  labels/
    generator.py       # Тройная и бинарная разметка
  models/
    backtest.py        # TimeSeriesWalkForwardSplitter
    baseline.py        # Логистическая регрессия (Baseline)
    train.py           # LightGBM-эксперимент + Optuna тюнинг
    predictor.py       # Инференс на новых свечах (двусторонний)
  paper_trading/
    engine.py          # Двусторонний движок виртуальной торговли
  handlers/            # admin/, user/ — обработчики кнопок и команд
  keyboards/           # user.py — динамические клавиатуры
tests/
  test_features.py     # Тесты индикаторов
  test_labels.py       # Математический тест на отсутствие утечки данных
  test_paper.py        # Тесты Long/Short сделок и лимитов объемов
  test_optuna.py       # Тест сеточного поиска параметров
  test_rollback.py     # Тест SRE-автоотката при просадках
```

---

## 🚀 Быстрый старт

### Локально

```bash
git clone <repo_url>
cd binance_quant_bot

cp .env.example .env
# Заполните токен бота и ID администраторов в .env

pip install -r requirements.txt
pip install optuna

# Применить миграции базы данных
alembic upgrade head

# Собрать историю BTC, ETH и SOL (без этого запуск не имеет смысла)
python -m scripts.backfill --symbol BTC/USDT --timeframe 1h --days 90
python -m scripts.backfill --symbol ETH/USDT --timeframe 1h --days 90
python -m scripts.backfill --symbol SOL/USDT --timeframe 1h --days 90

# Запуск бота (требуется запущенный Redis)
python -m src.main
```

### Скрипт калибровки SL/TP рисков
Для подбора наиболее математически оптимальных параметров стоп-лосса и тейк-профита по конкретной монете на основе обученной модели запустите:
```bash
python -m scripts.calibrate --symbol SOL/USDT --timeframe 1h
```
Скрипт выдаст отчет о коэффициенте Шарпа, матожидании сделки и порекомендует значения для записи в `.env`.

---

## 🔧 Конфигурация (.env)

```env
BOT_TOKEN=your_bot_token_here
ADMIN_IDS=[123456789,987654321]

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Модель и обучение
TARGET_COL=target_triple

# Риск-менеджмент сделок (Paper Trading)
PAPER_SL_PCT=0.02              # Дефолтный стоп-лосс (2%)
PAPER_TP_PCT=0.04              # Дефолтный тейк-профит (4%)
PAPER_RISK_PCT=0.10            # Доля капитала на сделку (10% от общего баланса)
PAPER_MIN_ALLOCATION=10.0      # Минимальный размер сделки в USD

# Настройка Optuna тюнинга
OPTUNA_TUNING_ENABLED=false     # Включить/выключить подбор параметров при переобучении
OPTUNA_TRIALS=15               # Количество итераций подбора

# Метрики SRE автоотката при деградации
ROLLBACK_CHECK_WINDOW=10       # Окно сделок для анализа качества
ROLLBACK_WIN_RATE_THRESHOLD=0.35 # Минимальный Win Rate
ROLLBACK_MAX_DRAWDOWN_THRESHOLD=0.15 # Максимально допустимая просадка (15%)

# Nexus SRE (опционально)
NEXUS_APP_SECRET=your_nexus_app_secret_here
NEXUS_ENDPOINT_URL=http://nexus-webhook:8000/events/app
NEXUS_PROJECT_NAME=binance_quant_bot
```

---

## ⚡ Стек

| Компонент | Технология |
|---|---|
| Интерфейс | Aiogram 3 |
| Котировки | CCXT (Binance) |
| База данных | SQLite + SQLAlchemy (async) |
| Миграции | Alembic |
| Кэш и FSM | Redis 7 |
| Модели ИИ | scikit-learn, LightGBM |
| Оптимизация ИИ | Optuna |
| Сборка данных | Pandas, PyArrow (Parquet) |
| Тесты | Pytest, Pytest-asyncio |
| Деплой | GitHub GCR → Docker Compose → SSH VPS |

---

## ⚠️ Текущие ограничения / TODO
* Реальные ордера на биржу не отправляются — система полностью безопасна и работает только в режиме симуляции (Paper Trading).
* Раз в сутки автокалибровка рисков во время `retrain_loop` отправляет отчет админам, но параметры применяются динамически «в памяти». Для сохранения между перезапусками рекомендуется прописывать новые параметры в `.env`.
