```markdown
# 🏛 Архитектурная документация MarketMind v1.0

MarketMind — это гибридная система для алгоритмической торговли и исследований криптовалютных фьючерсов. Архитектура четко разделена на два изолированных мира
1. Trading Core (247) Легковесный асинхронный процесс исполнения ордеров и риск-менеджмента.
2. Research Lab (On-Demand) Изолированная среда исследований, построенная вокруг Концепции Эксперимента (Purged Walk-Forward CV, MLOps-артефакты, Economic Quality Gate).

---

## 🎯 Ключевые архитектурные принципы

1. Эксперимент как Главная Сущность (Experiment-Centric)
    Результат любого исследования — это не просто файл `.pkl`, а неизменяемый каталог артефактов в `artifactsexperimentsexp_XXXX` (конфиг, хэш датасета, OOS-предсказания, сделки, метрики, визуальный `report.md` и модель).
    Исключена проблема хаоса версий моделей (`model_final_v2_REAL.pkl`).

2. Единый источник правды для фичей (Single Source of Truth)
    Модуль `srcstrategyfeatures.py` является единственным местом расчета технических индикаторов.
    И исследовательский конвейер (`research`), и боевой бот (`src`) импортируют фичи из одного места, исключая разрыв между тестом и продакшеном (Research-Production Skew).

3. Purged Walk-Forward Validation (Защита от утечек)
    Запрещено обучать модели без кросс-валидации со временным сдвигом.
    Между тренировочными и тестовыми окнами обязателен Purge Gap (зазор), равный горизонту позиции, для исключения заглядывания в будущее.

4. Изолированный Deployment Gate (Promote)
    Исследовательский пайплайн завершается созданием артефакта эксперимента.
    Промоушен модели в продакшен выполняется отдельным скриптом-цензором (`scriptspromote_candidate.py`) на основе проверки OOS-метрик против регламента рисков.

---

## 🔄 Двухконтурная Схема Системы

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. RESEARCH LAB (On-Demand  Experiment-Centric)                       │
│                                                                         │
│  dataset.py ──► labeling.py ──► validation.py ──► train.py ──► backtest │
│       │                                                            │    │
│  (импортирует)                                                     ▼    │
│  srcstrategyfeatures.py ──► report.py ──► artifactsexperimentsexp_N │
└─────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼ (scriptspromote_candidate.py)
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. TRADING CORE (Single Async Process, 247)                            │
│                                                                         │
│  [MarketService] ──► [StrategyService] ──► [RiskService] ──► [Execution]│
│        │                   │                    │                │      │
│        ▼                   ▼                    ▼                ▼      │
│   [SQLite Kline]  [modelsactive.pkl]     [RiskGuard]    [SQLite Trade]
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄 Хранилище Экспериментов (`artifactsexperimentsexp_XXXX`)

Каждый запуск исследований формирует неизменяемую директорию

```text
exp_0042
├── config.yaml          # Зафиксированная конфигурация эксперимента
├── metrics.json         # OOS-метрики (Profit Factor, WinRate, Sharpe, MaxDD)
├── model.pkl            # Обученная модель
├── predictions.parquet  # Честные Out-of-Sample прогнозы по фолдам
├── trades.parquet       # Лог всех симулированных сделок с учетом издержек
└── report.md            # Сгенерированный текстовый и графический отчет
```

---

## 🛠 Руководство разработчика

### Запуск эксперимента
```bash
python -m scripts.run_experiment --config researchconfigslgbm_btc_1h.yaml
```

### Проверка Quality Gate и продвижение модели в Продакшен
```bash
python -m scripts.promote_candidate --exp-id exp_0042
```

### Запуск тестов
```bash
python -m pytest tests
```
```