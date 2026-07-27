```markdown
# 📂 Структура Проекта MarketMind v1.0

```text
MarketMind
├── .github
│   └── workflows
│       └── deploy.yml            # CICD pytest, сборка Docker и SSH-деплой
│
├── artifacts                    # Неизменяемый архив экспериментов (в .gitignore)
│   └── experiments
│       └── exp_0001             # Артефакты конкретного эксперимента
│           ├── config.yaml
│           ├── metrics.json
│           ├── model.pkl
│           ├── predictions.parquet
│           ├── trades.parquet
│           └── report.md
│
├── models                       # Директория активной боевой модели
│   └── active.pkl                # Модель, используемая Trading Core в реальном времени
│
├── nexus_sdk                    # SRE Мониторинг SDK
│   ├── __init__.py
│   └── error_handler.py          # Heartbeat, HMAC подпись, перехват ошибок
│
├── research                     # ИССЛЕДОВАТЕЛЬСКАЯ ЛАБОРАТОРИЯ (Research Lab)
│   ├── backtest.py               # Симулятор OOS-торговли с учетом издержек
│   ├── dataset.py                # Загрузка, очистка klines + импорт srcstrategyfeatures.py
│   ├── experiment.py             # Создатель атомарной папки эксперимента
│   ├── labeling.py               # Генерация целевых переменных (таргетов)
│   ├── report.py                 # Генератор отчетности (report.md, графики)
│   ├── train.py                  # Адаптер ML-моделей (LightGBM, CatBoost)
│   └── validation.py             # Purged Walk-Forward Cross-Validation
│
├── scripts                      # CLI скрипты управления
│   ├── backfill.py               # Выкачивание истории свечей с Binance в SQLite
│   ├── promote_candidate.py      # DEPLOYMENT GATE проверка Quality Gate и замена active.pkl
│   └── run_experiment.py         # Единый запуск исследования
│
├── src                          # ТОРГОВЫЙ ДВИЖОК (Trading Core — 247)
│   ├── bot                      # Telegram-интерфейс (Aiogram 3)
│   │   ├── handlers.py
│   │   └── keyboards.py
│   │
│   ├── exchange                 # Адаптеры биржи (CCXT Изоляция)
│   │   ├── base.py
│   │   ├── binance.py
│   │   └── paper.py
│   │
│   ├── risk                     # Модуль Управления Рисками
│   │   ├── guards.py             # RiskGuard (Circuit Breaker, Daily Loss)
│   │   └── sizer.py              # Position Sizer и ATRpct стопы
│   │
│   ├── services                 # Исполнители событий (Event Handlers)
│   │   ├── execution_service.py
│   │   ├── market.py
│   │   ├── notifier_service.py
│   │   ├── risk_service.py
│   │   └── strategy_service.py
│   │
│   ├── strategy                 # Математика и Инференс
│   │   ├── features.py           # ⚠️ ЕДИНСТВЕННЫЙ ИСТОЧНИК ПРАВДЫ ДЛЯ ФИЧЕЙ
│   │   ├── generator.py
│   │   └── model.py              # Predictor (Легкий инференс active.pkl)
│   │
│   ├── config.py
│   ├── db.py                     # SQLAlchemy 2.0 Async + SQLite WAL
│   ├── event_bus.py              # Легковесный AsyncEventBus
│   └── main.py                   # Точка входа боевого бота
│
├── tests                        # Pytest тесты
├── Dockerfile
├── docker-compose.yml
├── ARCHITECTURE.md
├── DEPENDENCY_RULES.md
├── PROJECT_TREE.md
├── README.md
├── ROADMAP.md
└── STANDARD.md
```
```