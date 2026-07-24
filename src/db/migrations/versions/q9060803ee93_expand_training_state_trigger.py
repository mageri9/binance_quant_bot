"""expand training state trigger

Revision ID: q9060803ee93
Revises: p9060803ee92
"""
from alembic import op
import sqlalchemy as sa


revision = "q9060803ee93"
down_revision = "p9060803ee92"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "training_states",
        "last_trigger",
        existing_type=sa.String(length=40),
        type_=sa.String(length=100),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "training_states",
        "last_trigger",
        existing_type=sa.String(length=100),
        type_=sa.String(length=40),
        existing_nullable=True,
    )
