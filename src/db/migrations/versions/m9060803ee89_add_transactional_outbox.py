"""add transactional outbox event core

Revision ID: m9060803ee89
Revises: k9060803ee87
Create Date: 2026-07-23 02:25:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "m9060803ee89"
down_revision: Union[str, None] = "k9060803ee87"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.rename_table("exchange_fills", "fills")
    op.create_table(
        "exchange_orders",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("order_intent_id", sa.Integer(), sa.ForeignKey("order_intents.id")),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column("binance_order_id", sa.String(length=64), nullable=False),
        sa.Column("client_order_id", sa.String(length=36)),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("raw_payload", sa.JSON()),
        sa.Column("exchange_update_time", sa.BigInteger()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("environment", "binance_order_id", name="uq_exchange_order_environment_binance_id"),
    )
    op.create_index("ix_exchange_orders_environment", "exchange_orders", ["environment"])
    op.create_index("ix_exchange_orders_client_order_id", "exchange_orders", ["client_order_id"])
    op.create_index("ix_exchange_orders_symbol", "exchange_orders", ["symbol"])
    op.create_index("ix_exchange_orders_status", "exchange_orders", ["status"])
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(length=36), nullable=False, unique=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("correlation_id", sa.String(length=36), nullable=False),
        sa.Column("causation_id", sa.String(length=36)),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("binance_event_id", sa.String(length=160), unique=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    for column in ("event_type", "correlation_id", "causation_id", "occurred_at", "published_at"):
        op.create_index(f"ix_outbox_events_{column}", "outbox_events", [column])
    op.create_table(
        "processed_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("consumer", sa.String(length=64), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("consumer", "event_id", name="uq_processed_event_consumer_event"),
    )
    op.create_index("ix_processed_events_consumer", "processed_events", ["consumer"])
    op.create_table(
        "model_deployments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("artifact_uri", sa.Text()),
        sa.Column("metrics", sa.JSON()),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_model_deployments_model_id", "model_deployments", ["model_id"])
    op.create_index("ix_model_deployments_status", "model_deployments", ["status"])


def downgrade() -> None:
    op.drop_table("model_deployments")
    op.drop_table("processed_events")
    op.drop_table("outbox_events")
    op.drop_table("exchange_orders")
    op.rename_table("fills", "exchange_fills")
