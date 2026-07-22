"""add execution event inbox and environment separation

Revision ID: k9060803ee87
Revises: j9060803ee86
Create Date: 2026-07-23 00:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "k9060803ee87"
down_revision: Union[str, None] = "j9060803ee86"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("environment", sa.String(length=20), server_default="paper", nullable=False),
    )
    op.create_index("ix_trades_environment", "trades", ["environment"])

    op.add_column(
        "order_intents",
        sa.Column("purpose", sa.String(length=24), server_default="ENTRY", nullable=False),
    )
    op.add_column("order_intents", sa.Column("parent_intent_id", sa.Integer()))
    op.add_column(
        "order_intents",
        sa.Column("reduce_only", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column("order_intents", sa.Column("exchange_update_time", sa.BigInteger()))

    op.create_table(
        "balance_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("asset", sa.String(length=16), nullable=False),
        sa.Column("wallet_balance", sa.Numeric(38, 18), nullable=False),
        sa.Column("available_balance", sa.Numeric(38, 18)),
        sa.Column("cross_wallet_balance", sa.Numeric(38, 18)),
        sa.Column("update_time", sa.BigInteger()),
        sa.Column("raw_payload", sa.JSON()),
        sa.Column("reconciled_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("environment", "asset", name="uq_balance_environment_asset"),
    )
    op.create_index("ix_balance_snapshots_environment", "balance_snapshots", ["environment"])
    op.create_index("ix_balance_snapshots_asset", "balance_snapshots", ["asset"])

    op.create_table(
        "exchange_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_key", sa.String(length=160), nullable=False, unique=True),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column("event_time", sa.BigInteger()),
        sa.Column("received_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_exchange_events_environment", "exchange_events", ["environment"])
    op.create_index("ix_exchange_events_event_type", "exchange_events", ["event_type"])


def downgrade() -> None:
    op.drop_table("exchange_events")
    op.drop_table("balance_snapshots")
    op.drop_column("order_intents", "exchange_update_time")
    op.drop_column("order_intents", "reduce_only")
    op.drop_column("order_intents", "parent_intent_id")
    op.drop_column("order_intents", "purpose")
    op.drop_index("ix_trades_environment", table_name="trades")
    op.drop_column("trades", "environment")
