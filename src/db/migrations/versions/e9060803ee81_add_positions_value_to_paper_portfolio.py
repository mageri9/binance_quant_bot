"""add positions_value to paper portfolio

Revision ID: e9060803ee81
Revises: d9060803ee80
Create Date: 2026-07-14 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e9060803ee81'
down_revision: Union[str, None] = 'd9060803ee80'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Используем batch_alter_table для безопасного добавления NOT NULL колонки в SQLite
    with op.batch_alter_table('paper_portfolios') as batch_op:
        batch_op.add_column(
            sa.Column('positions_value', sa.Float(), nullable=False, server_default='0.0')
        )


def downgrade() -> None:
    # Безопасное удаление колонки в SQLite
    with op.batch_alter_table('paper_portfolios') as batch_op:
        batch_op.drop_column('positions_value')