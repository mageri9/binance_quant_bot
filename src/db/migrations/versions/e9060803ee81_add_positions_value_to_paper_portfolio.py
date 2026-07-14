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
    # Добавляем поле для хранения стоимости открытых позиций
    op.add_column(
        'paper_portfolios',
        sa.Column('positions_value', sa.Float(), nullable=False, server_default='0.0')
    )
    # Для существующих записей вычисляем значение из cash (так как позиций нет)
    op.execute("""
        UPDATE paper_portfolios 
        SET positions_value = 0.0
    """)


def downgrade() -> None:
    op.drop_column('paper_portfolios', 'positions_value')