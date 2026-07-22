"""add live execution ledger

Revision ID: j9060803ee86
Revises: i9060803ee85
Create Date: 2026-07-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "j9060803ee86"
down_revision: Union[str, None] = "i9060803ee85"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("source", sa.String(length=20), server_default="paper", nullable=False),
    )
    op.add_column("trades", sa.Column("client_order_id", sa.String(length=36)))
    op.add_column("trades", sa.Column("entry_order_id", sa.String(length=64)))
    op.add_column("trades", sa.Column("exit_order_id", sa.String(length=64)))
    op.add_column("trades", sa.Column("model_id", sa.String(length=100)))
    op.add_column("trades", sa.Column("last_reconciled_at", sa.DateTime(timezone=True)))
    op.create_index("ix_trades_client_order_id", "trades", ["client_order_id"])

    op.create_table(
        "order_intents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("correlation_id", sa.String(length=36), nullable=False, unique=True),
        sa.Column("client_order_id", sa.String(length=36), nullable=False, unique=True),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("order_type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=24), server_default="PENDING", nullable=False),
        sa.Column("requested_amount", sa.Numeric(38, 18), nullable=False),
        sa.Column("requested_price", sa.Numeric(38, 18)),
        sa.Column("filled_amount", sa.Numeric(38, 18)),
        sa.Column("average_fill_price", sa.Numeric(38, 18)),
        sa.Column("commission", sa.Numeric(38, 18)),
        sa.Column("commission_asset", sa.String(length=16)),
        sa.Column("exchange_order_id", sa.String(length=64)),
        sa.Column("raw_status", sa.String(length=32)),
        sa.Column("model_id", sa.String(length=100)),
        sa.Column("prediction_id", sa.Integer(), sa.ForeignKey("prediction_logs.id")),
        sa.Column("trade_id", sa.Integer(), sa.ForeignKey("trades.id")),
        sa.Column("sl_price", sa.Numeric(38, 18)),
        sa.Column("tp_price", sa.Numeric(38, 18)),
        sa.Column("raw_response", sa.JSON()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True)),
        sa.Column("filled_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_order_intents_environment", "order_intents", ["environment"])
    op.create_index("ix_order_intents_symbol", "order_intents", ["symbol"])
    op.create_index("ix_order_intents_status", "order_intents", ["status"])
    op.create_index("ix_order_intents_exchange_order_id", "order_intents", ["exchange_order_id"])

    op.create_table(
        "exchange_fills",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("fill_key", sa.String(length=128), nullable=False, unique=True),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("exchange_trade_id", sa.String(length=64)),
        sa.Column("exchange_order_id", sa.String(length=64)),
        sa.Column("client_order_id", sa.String(length=36)),
        sa.Column("price", sa.Numeric(38, 18), nullable=False),
        sa.Column("amount", sa.Numeric(38, 18), nullable=False),
        sa.Column("commission", sa.Numeric(38, 18), server_default="0", nullable=False),
        sa.Column("commission_asset", sa.String(length=16)),
        sa.Column("realized_pnl", sa.Numeric(38, 18)),
        sa.Column("exchange_time", sa.BigInteger()),
        sa.Column("raw_payload", sa.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_exchange_fills_environment", "exchange_fills", ["environment"])
    op.create_index("ix_exchange_fills_symbol", "exchange_fills", ["symbol"])

    op.create_table(
        "position_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=8)),
        sa.Column("amount", sa.Numeric(38, 18), nullable=False),
        sa.Column("entry_price", sa.Numeric(38, 18)),
        sa.Column("mark_price", sa.Numeric(38, 18)),
        sa.Column("unrealized_pnl", sa.Numeric(38, 18)),
        sa.Column("leverage", sa.Numeric(18, 8)),
        sa.Column("exchange_update_time", sa.BigInteger()),
        sa.Column("raw_payload", sa.JSON()),
        sa.Column("reconciled_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("environment", "symbol", name="uq_position_environment_symbol"),
    )
    op.create_index("ix_position_snapshots_environment", "position_snapshots", ["environment"])
    op.create_index("ix_position_snapshots_symbol", "position_snapshots", ["symbol"])

    op.create_table(
        "reconciliation_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("actions", sa.JSON()),
        sa.Column("details", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_reconciliation_runs_environment", "reconciliation_runs", ["environment"])


def downgrade() -> None:
    op.drop_table("reconciliation_runs")
    op.drop_table("position_snapshots")
    op.drop_table("exchange_fills")
    op.drop_table("order_intents")
    op.drop_index("ix_trades_client_order_id", table_name="trades")
    op.drop_column("trades", "last_reconciled_at")
    op.drop_column("trades", "model_id")
    op.drop_column("trades", "exit_order_id")
    op.drop_column("trades", "entry_order_id")
    op.drop_column("trades", "client_order_id")
    op.drop_column("trades", "source")
