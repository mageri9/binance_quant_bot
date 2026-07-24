"""add prediction log diagnostics

Revision ID: r9060803ee94
Revises: q9060803ee93
"""
from alembic import op
import sqlalchemy as sa


revision = "r9060803ee94"
down_revision = "q9060803ee93"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("prediction_logs", sa.Column("reason", sa.Text(), nullable=True))
    op.add_column("prediction_logs", sa.Column("details", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("prediction_logs", "details")
    op.drop_column("prediction_logs", "reason")
