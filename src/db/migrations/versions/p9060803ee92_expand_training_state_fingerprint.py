"""expand training state dataset fingerprint

Revision ID: p9060803ee92
Revises: o9060803ee91
"""
from alembic import op
import sqlalchemy as sa


revision = "p9060803ee92"
down_revision = "o9060803ee91"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "training_states",
        "last_dataset_fingerprint",
        existing_type=sa.String(length=40),
        type_=sa.String(length=64),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "training_states",
        "last_dataset_fingerprint",
        existing_type=sa.String(length=64),
        type_=sa.String(length=40),
        existing_nullable=True,
    )
