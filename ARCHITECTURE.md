# Архитектура

Один асинхронный процесс. Никакого отдельного Research Lab, никакого
каталога `artifacts/experiments/exp_XXXX`, никакого `promote_candidate.py` —
это были нереализованные планы предыдущей версии документации и в коде их
нет. Обучение модели — это `scripts/train.py`, точка. Если понадобится
больше — добавим тогда же, когда понадобится, не раньше.

## Поток событий

```
[Binance / Paper Exchange]
        │
        ▼
MarketService ──CandleClosedEvent──► StrategyService
                                          │
                                    SignalEmittedEvent
                                          ▼
                                     RiskService
                                          │
                                    OrderApprovedEvent
                                          ▼
                                   ExecutionService
                                          │
                              OrderExecutedEvent / TradeClosedEvent
                                          ▼
                                   NotifierService (Telegram + Nexus SDK)
```

Всё общение между сервисами — через `AsyncEventBus` (`src/event_bus.py`),
in-memory, без Redis/Kafka. Это сознательное ограничение под ВПС 2 ГБ RAM,
не временная заглушка.

## Хранилище

SQLite (WAL mode) — таблицы `klines`, `trades`, `prediction_logs`.
Модель — один файл `models/saved_models/lgbm_<symbol>_<timeframe>.pkl`,
пересоздаётся вручную через `scripts/train.py`. Никакого versioning-реестра
моделей не требуется на этом масштабе.

## Три режима биржи

- `paper` — `PaperExchange`, локальная симуляция, источник честной OOS-проверки
  (см. `ROADMAP.md`, шаг 5).
- `testnet` / `mainnet` — `BinanceExchange` через `ccxt`.

## Модель сигнала

`src/strategy/features.py` — единственное место расчёта индикаторов
(RSI, MACD, Bollinger, ATR, ADX). `src/strategy/model.py` — LightGBM-регрессор
ожидаемой доходности long/short. `src/strategy/generator.py` — при отсутствии
файла модели откатывается на простое RSI/MACD правило. Никакого отдельного
слоя абстракции под несколько типов моделей не нужно, пока используется
ровно один тип.

## Границы модулей

Правила из старого `DEPENDENCY_RULES.md` остаются в силе и уже соблюдены
кодом: `strategy/` не знает про биржу и БД, `risk/` не знает про SQLite,
`exchange/` не знает про стратегию и риск. Прокидывать это в отдельный
документ смысла нет — правила видны из самих импортов.