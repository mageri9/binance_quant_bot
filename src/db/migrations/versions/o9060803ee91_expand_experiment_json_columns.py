"""expand experiment JSON text columns

Revision ID: o9060803ee91
Revises: n9060803ee90
"""
from alembic import op
import sqlalchemy as sa


revision = "o9060803ee91"
down_revision = "n9060803ee90"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("experiments", "parameters", type_=sa.Text())
    op.alter_column("experiments", "metrics", type_=sa.Text())


def downgrade() -> None:
    op.alter_column("experiments", "metrics", type_=sa.String(length=500))
    op.alter_column("experiments", "parameters", type_=sa.String(length=500))
