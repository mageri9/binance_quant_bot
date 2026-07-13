from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, String, func, UniqueConstraint
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
    parameters: Mapped[str] = mapped_column(String(500), nullable=False)  # JSON-строка с параметрами
    metrics: Mapped[str] = mapped_column(String(500), nullable=False)     # JSON-строка с метриками точности
    git_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Experiment id={self.id} model={self.model_name} metrics={self.metrics}>"


class PaperPortfolio(Base):
    __tablename__ = "paper_portfolios"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    balance: Mapped[float] = mapped_column(default=10000.0, nullable=False)
    cash: Mapped[float] = mapped_column(default=10000.0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(10), default="OPEN", nullable=False)  # "OPEN" или "CLOSED"
    entry_price: Mapped[float] = mapped_column(nullable=False)
    entry_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    exit_price: Mapped[float] = mapped_column(nullable=True)
    exit_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    amount: Mapped[float] = mapped_column(nullable=False)  # купленное количество монет
    sl_price: Mapped[float] = mapped_column(nullable=True)
    tp_price: Mapped[float] = mapped_column(nullable=True)
    pnl: Mapped[float] = mapped_column(nullable=True)      # профит / лосс в долларах
    entry_candle_time: Mapped[int] = mapped_column(BigInteger, nullable=False)

    def __repr__(self) -> str:
        return f"<PaperTrade symbol={self.symbol} status={self.status} pnl={self.pnl}>"