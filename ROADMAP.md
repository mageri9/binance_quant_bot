```markdown
# 🚀 Production Roadmap v1.0 (Experiment-Centric)

План поэтапной реализации системы с упором на строгость исследовательского конвейера и изоляцию торгового ядра.

---

### Фаза 1 Базовый каркас, БД и Биржа (Готово)
 Цель Изолированные адаптеры исполнения и база данных.

- [x] 1.1. Конфигурация и БД (`srcconfig.py`, `srcdb.py`)
  - Pydantic Settings, SQLAlchemy Async (SQLite WAL mode).
  - Модели `Kline`, `Trade`, `PredictionLog`.
- [x] 1.2. Биржевые Адаптеры (`srcexchange`)
  - `BaseExchange`, `PaperExchange` и `BinanceExchange` (CCXT Futures + Algo Order API).
- [x] 1.3. Шина Событий (`srcevent_bus.py`)
  - Легковесный `AsyncEventBus` и типы событий.

---

### Фаза 2 Торговый Движок (Trading Core 247) (Готово)
 Цель Линейный пайплайн обработки рыночных событий.

- [x] 2.1. Единый модуль индикаторов (`srcstrategyfeatures.py`)
  - Расчет RSI, ATR, MACD, ADX, Bollinger Bands в `float32`.
- [x] 2.2. Модуль Рисков (`srcrisk`)
  - `RiskGuard` (Circuit Breaker, дневной лимит просадки, лимит позиций).
  - `Position Sizer` (размер лота от депозита, расчет стопов).
- [x] 2.3. Исполнительные сервисы (`srcservices`)
  - `MarketService`, `StrategyService`, `RiskService`, `ExecutionService`.

---

### Фаза 3 Telegram Бот и SRE Мониторинг (Готово)
 Цель Управление, алерты и устойчивость.

- [x] 3.1. Telegram Бот (`srcbot`)
  - Aiogram 3.x с командами `status`, `positions`, `trades`, `risk`.
- [x] 3.2. SRE Notifier (`srcservicesnotifier_service.py`, `nexus_sdk`)
  - Telegram алерты + Nexus SDK (Heartbeat 15с, перехват ошибок).

---

### Фаза 4 Исследовательский конвейер (Research Lab) (ТЕКУЩАЯ)
 Цель Модульная лаборатория экспериментов с гарантией отсутствия утечек данных.

- [ ] 4.1. Модули конвейера (`research`)
  - `dataset.py` Сбор OHLCV + очистка gapstz + подключение `srcstrategyfeatures.py`.
  - `labeling.py` Изолированный расчет таргетов (Post-cost return, Triple Barrier).
  - `validation.py` Purged Walk-Forward Splitter (зазор = `horizon`).
  - `train.py` Адаптер моделей (LightGBM, CatBoost).
  - `backtest.py` Экономический симулятор (комиссии 0.08%, проскальзывание 0.04%).
  - `report.py` Генератор `report.md` и метрик.
  - `experiment.py` Создатель неизменяемого каталога `artifactsexperimentsexp_XXXX`.
- [ ] 4.2. Оркестратор (`scriptsrun_experiment.py`)
  - CLI запуск эксперимента по `.yaml` конфигу.

---

### Фаза 5 Deployment Gate и Деплой (Следующая)
 Цель Безопасное обновление моделей в продакшене и CICD.

- [ ] 5.1. Цензор промоушена (`scriptspromote_candidate.py`)
  - Проверка `exp_XXXXmetrics.json` против Quality Gate ($PF ge 1.15$, $Expectancy  0.001$).
  - Атомарная перезапись `modelsactive.pkl` и Hot-Reload в боте.
- [ ] 5.2. Упаковка и CICD (`Dockerfile`, `.githubworkflowsdeploy.yml`)
  - Docker сборка (`python3.12-slim` + `libgomp1`).
  - Деплой на VPS с установкой флага обслуживания в Redis.
```
