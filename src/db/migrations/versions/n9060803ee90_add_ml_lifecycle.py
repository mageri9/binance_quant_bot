"""add durable ML lifecycle state and registry metadata

Revision ID: n9060803ee90
Revises: m9060803ee89
"""
from alembic import op
import sqlalchemy as sa

revision = "n9060803ee90"
down_revision = "m9060803ee89"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name, type_, nullable in (
        ("symbol", sa.String(20), False), ("timeframe", sa.String(10), False),
        ("target", sa.String(50), False), ("parameters", sa.JSON(), True),
        ("feature_schema", sa.JSON(), True), ("dataset_fingerprint", sa.String(64), True),
        ("offline_metrics", sa.JSON(), True), ("live_metrics", sa.JSON(), True),
        ("trading_metrics", sa.JSON(), True), ("trained_at", sa.DateTime(timezone=True), True),
        ("promoted_at", sa.DateTime(timezone=True), True),
        ("shadow_started_at", sa.DateTime(timezone=True), True),
    ):
        op.add_column("model_deployments", sa.Column(name, type_, nullable=nullable, server_default="unknown" if name in {"symbol", "timeframe", "target"} else None))
    op.drop_column("model_deployments", "metrics")
    op.create_index("ix_model_deployments_symbol", "model_deployments", ["symbol"])
    op.create_index("ix_model_deployments_dataset_fingerprint", "model_deployments", ["dataset_fingerprint"])
    op.create_table("training_states",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False), sa.Column("timeframe", sa.String(10), nullable=False),
        sa.Column("target", sa.String(50), nullable=False), sa.Column("last_trained_candle", sa.BigInteger()),
        sa.Column("last_dataset_fingerprint", sa.String(64)), sa.Column("last_trained_at", sa.DateTime(timezone=True)),
        sa.Column("last_trigger", sa.String(40)), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("symbol", "timeframe", "target", name="uq_training_scope"))
    op.create_index("ix_training_states_symbol", "training_states", ["symbol"])
    for name, type_, nullable, default in (
        ("timeframe", sa.String(10), False, "unknown"), ("candle_time", sa.BigInteger(), True, None),
        ("horizon", sa.Integer(), False, "5"), ("resolved_at", sa.DateTime(timezone=True), True, None),
        ("outcome_price", sa.Float(), True, None), ("true_label", sa.Integer(), True, None),
        ("realized_return", sa.Float(), True, None)):
        op.add_column("prediction_logs", sa.Column(name, type_, nullable=nullable, server_default=default))
    op.create_index("ix_prediction_logs_candle_time", "prediction_logs", ["candle_time"])
    op.create_index("ix_prediction_logs_resolved_at", "prediction_logs", ["resolved_at"])


def downgrade() -> None:
    op.drop_table("training_states")
    for name in ("timeframe", "candle_time", "horizon", "resolved_at", "outcome_price", "true_label", "realized_return"):
        op.drop_column("prediction_logs", name)
    for name in ("symbol", "timeframe", "target", "parameters", "feature_schema", "dataset_fingerprint", "offline_metrics", "live_metrics", "trading_metrics", "trained_at", "promoted_at", "shadow_started_at"):
        op.drop_column("model_deployments", name)
    op.add_column("model_deployments", sa.Column("metrics", sa.JSON()))
