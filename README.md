# MarketMind

MarketMind — исследовательская платформа для алгоритмической торговли и машинного обучения. Система собирает свечные данные Binance, строит признаки, обучает и версионирует модели, выполняет walk-forward-бэктесты и оценивает стратегии в paper/shadow режимах.

> Проект предназначен для исследований и тестовой торговли. Режим `mainnet` может отправлять реальные ордера и требует отдельной проверки конфигурации и рисков.

## Возможности

- асинхронный Telegram-бот на Aiogram;
- сбор и хранение OHLCV-данных Binance через CCXT;
- признаки RSI, MACD, Bollinger Bands, ATR, ADX, волатильность и отношение объёма;
- бинарная и трёхклассовая классификация (`LONG`, `HOLD`, `SHORT`);
- walk-forward-валидация, бэктест и Optuna-калибровка SL/TP;
- paper trading с long/short позициями, комиссиями, проскальзыванием и журналом сделок;
- реестр моделей, датасетов и экспериментов, OOS-артефакты и контроль дрейфа;
- автоматическое переобучение с экономическим quality gate и возможностью rollback;
- Redis для FSM и rate limiting, SQLite по умолчанию или PostgreSQL.

## Архитектура

```text
Binance -> collector -> database -> feature engineering -> labels
                                      |
                                      v
                         training / backtest / calibration
                                      |
                                      v
                         model registry -> predictor
                                      |
                                      v
                         risk engine -> paper/shadow execution
                                      |
                                      v
                              Telegram notifications
```

Основной код находится в `src/`: `data` отвечает за сбор данных, `features` — за признаки, `models` — за обучение и прогнозы, `risk` и `exchange` — за риск и исполнение, `handlers` — за Telegram-интерфейс. Вспомогательные операции находятся в `scripts/`, тесты — в `tests/`.

## Требования

- Python 3.12 или Docker;
- Redis 7+;
- для PostgreSQL: PostgreSQL 14+ и драйвер из `requirements.txt`;
- Telegram Bot Token. Ключи Binance нужны только для `testnet` или `mainnet`.

## Быстрый старт локально

```bash
git clone <url-репозитория>
cd MarketMind
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Создайте `.env` на основе `.env.example` и задайте минимум:

```dotenv
BOT_TOKEN=123456:telegram-token
ADMIN_IDS=[123456789]
TRADING_MODE=paper
REDIS_HOST=localhost
```

В режиме `paper` ключи Binance не обязательны. Для `testnet`/`mainnet` укажите `BINANCE_API_KEY` и `BINANCE_API_SECRET`; для боевой торговли используйте отдельные ключи с минимально необходимыми разрешениями.

Примените миграции и запустите бота:

```bash
alembic upgrade head
python -m src.main
```

По умолчанию используется SQLite в `src/db/db.db`. Путь можно изменить через `DATABASE_URL`.

## Запуск через Docker Compose

```bash
copy .env.example .env       # Windows
# cp .env.example .env      # Linux/macOS
docker compose up -d
docker compose logs -f bot
```

Compose запускает приложение и Redis. Файлы базы, логов и моделей монтируются из каталогов проекта. Если используется внешний PostgreSQL/сервис Nexus, подключите соответствующие внешние Docker-сети, указанные в `docker-compose.yml`, либо удалите их из compose-файла для локального запуска.

## Полезные команды

Загрузить исторические свечи:

```bash
python -m scripts.backfill --symbol BTC/USDT --timeframe 1h --days 90
```

Запустить калибровку параметров риска:

```bash
python -m scripts.calibrate --symbol BTC/USDT --timeframe 1h
```

Мигрировать SQLite в PostgreSQL:

```bash
python scripts/migrate_sqlite_to_postgres.py --help
```

Проверить проект тестами:

```bash
pytest
```

## Конфигурация

Полный список переменных находится в `.env.example`. Наиболее важные:

| Переменная | Назначение | Значение по умолчанию |
| --- | --- | --- |
| `TRADING_MODE` | `paper`, `shadow`, `testnet` или `mainnet` | — |
| `ACTIVE_CONFIGS` | пары и таймфреймы для фоновых циклов | BTC, ETH, SOL / 1h |
| `MODEL_PATH` | путь к модели | `models/saved_models/lgbm_BTCUSDT_1h.pkl` |
| `DATABASE_URL` | PostgreSQL URL; пустое значение включает SQLite | пусто |
| `REDIS_HOST`, `REDIS_PORT` | подключение к Redis | `localhost:6379` |
| `RETRAIN_POLL_SECONDS` | период проверки переобучения | `900` |
| `PREDICTION_CONFIDENCE_THRESHOLD` | минимальная уверенность прогноза | `0.55` |

Секреты не коммитьте в Git. Файл `.env` должен оставаться локальным.

## Жизненный цикл модели

1. Collector сохраняет свечи в БД.
2. Feature engineering рассчитывает технические признаки, а генератор labels формирует целевую переменную.
3. Обучение сохраняет модель, метаданные датасета, параметры, Git SHA и метрики.
4. Walk-forward и OOS-бэктест проверяют качество на временных отложенных данных.
5. Калибровка подбирает параметры риска; новая модель регистрируется как challenger.
6. Экономический gate сравнивает её с production-моделью. Автопромоушен отключён по умолчанию (`MODEL_AUTO_PROMOTE_LEGACY=false`).

## Telegram-интерфейс

Пользовательские команды включают просмотр статуса, портфеля, позиций, сделок и моделей. Администраторы могут просматривать статистику, включать kill switch, сбрасывать риск-состояние и отправлять сообщения подписчикам. Список администраторов задаётся в `ADMIN_IDS`.

## Лицензия

Лицензия в репозитории пока не указана. Перед публичным распространением добавьте файл `LICENSE` и правила использования.
