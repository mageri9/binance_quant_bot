from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Float,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.core.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False, index=True)
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    is_subscribed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} user_id={self.user_id} username={self.username}>"


class Kline(Base):
    __tablename__ = "klines"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    open_time: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    open: Mapped[float] = mapped_column(nullable=False)
    high: Mapped[float] = mapped_column(nullable=False)
    low: Mapped[float] = mapped_column(nullable=False)
    close: Mapped[float] = mapped_column(nullable=False)
    volume: Mapped[float] = mapped_column(nullable=False)

    # Запрещаем дубликаты: для одной пары, таймфрейма и времени может быть только одна свеча
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "open_time", name="idx_symbol_tf_time"),
    )

    def __repr__(self) -> str:
        return f"<Kline symbol={self.symbol} tf={self.timeframe} time={self.open_time} close={self.close}>"


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(20), nullable=False)
    parameters: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string with training parameters.
    metrics: Mapped[str] = mapped_column(Text, nullable=False)     # JSON string with evaluation metrics.
    git_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Experiment id={self.id} model={self.model_name} metrics={self.metrics}>"


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    balance: Mapped[Decimal] = mapped_column(Numeric(38, 18), default=Decimal("10000"), nullable=False)
    cash: Mapped[Decimal] = mapped_column(Numeric(38, 18), default=Decimal("10000"), nullable=False)
    positions_value: Mapped[Decimal] = mapped_column(Numeric(38, 18), default=Decimal("0"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(10), default="OPEN", nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    entry_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    exit_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    sl_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    tp_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    entry_candle_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    is_short: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, server_default="0")
    timeout_candle_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source: Mapped[str] = mapped_column(
        String(20), default="paper", server_default="paper", nullable=False
    )
    # A paper projection and a live projection may coexist for the same symbol.
    environment: Mapped[str] = mapped_column(
        String(20), default="paper", server_default="paper", nullable=False, index=True
    )
    client_order_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    entry_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exit_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_reconciled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class OrderIntent(Base):
    """Durable intent written before any private exchange request."""

    __tablename__ = "order_intents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    correlation_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    environment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)
    purpose: Mapped[str] = mapped_column(
        String(24), default="ENTRY", server_default="ENTRY", nullable=False
    )
    parent_intent_id: Mapped[int | None] = mapped_column(
        ForeignKey("order_intents.id"), nullable=True
    )
    reduce_only: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), default="PENDING", server_default="PENDING", nullable=False, index=True
    )
    requested_amount: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    requested_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    filled_amount: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    average_fill_price: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18), nullable=True
    )
    commission: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    commission_asset: Mapped[str | None] = mapped_column(String(16), nullable=True)
    exchange_order_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    raw_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("prediction_logs.id"), nullable=True
    )
    trade_id: Mapped[int | None] = mapped_column(ForeignKey("trades.id"), nullable=True)
    sl_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    tp_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    raw_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exchange_update_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class ExchangeFill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fill_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    environment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    exchange_trade_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    client_order_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    price: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    commission: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), default=0, server_default="0", nullable=False
    )
    commission_asset: Mapped[str | None] = mapped_column(String(16), nullable=True)
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    exchange_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    raw_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PositionSnapshot(Base):
    __tablename__ = "position_snapshots"
    __table_args__ = (
        UniqueConstraint("environment", "symbol", name="uq_position_environment_symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    entry_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    mark_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    leverage: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    exchange_update_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    raw_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reconciled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class BalanceSnapshot(Base):
    """Exchange-reported futures balance; never derived from local trade cost."""

    __tablename__ = "balance_snapshots"
    __table_args__ = (
        UniqueConstraint("environment", "asset", name="uq_balance_environment_asset"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    asset: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    wallet_balance: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    available_balance: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    cross_wallet_balance: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    update_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    raw_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reconciled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExchangeEvent(Base):
    """Idempotent event inbox for the Binance User Data Stream."""

    __tablename__ = "exchange_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_key: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    environment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    event_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class ReconciliationRun(Base):
    __tablename__ = "reconciliation_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    actions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExchangeOrder(Base):
    """Latest exchange order projection, keyed by Binance's order identifier."""

    __tablename__ = "exchange_orders"
    __table_args__ = (
        UniqueConstraint("environment", "binance_order_id", name="uq_exchange_order_environment_binance_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_intent_id: Mapped[int | None] = mapped_column(ForeignKey("order_intents.id"), nullable=True)
    environment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    binance_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    client_order_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    raw_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    exchange_update_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class OutboxEvent(Base):
    """Append-only transactional outbox for in-process consumers and future transports."""

    __tablename__ = "outbox_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    correlation_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    causation_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    payload_version: Mapped[int] = mapped_column(default=1, nullable=False)
    binance_event_id: Mapped[str | None] = mapped_column(String(160), unique=True, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    attempts: Mapped[int] = mapped_column(default=0, server_default="0", nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ProcessedEvent(Base):
    """Consumer inbox used to make projections idempotent."""

    __tablename__ = "processed_events"
    __table_args__ = (UniqueConstraint("consumer", "event_id", name="uq_processed_event_consumer_event"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    consumer: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(36), nullable=False)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="unknown", index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, default="unknown")
    target: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown")
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_schema: Mapped[list | None] = mapped_column(JSON, nullable=True)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    offline_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    live_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trading_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    shadow_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TrainingState(Base):
    __tablename__ = "training_states"
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "target", name="uq_training_scope"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    target: Mapped[str] = mapped_column(String(50), nullable=False)
    last_trained_candle: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    last_dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_trigger: Mapped[str | None] = mapped_column(String(100), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    price: Mapped[float] = mapped_column(nullable=False)
    prediction: Mapped[int] = mapped_column(nullable=False)
    prob_short: Mapped[float] = mapped_column(nullable=False)
    prob_hold: Mapped[float] = mapped_column(nullable=False)
    prob_long: Mapped[float] = mapped_column(nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, default="unknown")
    candle_time: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    horizon: Mapped[int] = mapped_column(nullable=False, default=5)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    outcome_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    true_label: Mapped[int | None] = mapped_column(nullable=True)
    realized_return: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<PredictionLog symbol={self.symbol} pred={self.prediction} model={self.model_id}>"
