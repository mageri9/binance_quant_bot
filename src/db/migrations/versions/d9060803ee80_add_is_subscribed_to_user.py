"""add is_subscribed to user

Revision ID: d9060803ee80
Revises: c9060803ee7f
Create Date: 2026-07-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd9060803ee80'
down_revision: Union[str, None] = 'c9060803ee7f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Добавляем колонку users.is_subscribed с дефолтным значением True (1)
    op.add_column(
        'users',
        sa.Column('is_subscribed', sa.Boolean(), nullable=False, server_default=sa.text('1'))
    )


def downgrade() -> None:
    op.drop_column('users', 'is_subscribed')