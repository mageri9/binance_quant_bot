"""create klines table

Revision ID: a8060803ee7d
Revises:
Create Date: 2026-07-13 07:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a8060803ee7d'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'klines',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('open_time', sa.BigInteger(), nullable=False),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timeframe', 'open_time', name='idx_symbol_tf_time')
    )
    op.create_index(op.f('ix_klines_open_time'), 'klines', ['open_time'], unique=False)
    op.create_index(op.f('ix_klines_symbol'), 'klines', ['symbol'], unique=False)
    op.create_index(op.f('ix_klines_timeframe'), 'klines', ['timeframe'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_klines_timeframe'), table_name='klines')
    op.drop_index(op.f('ix_klines_symbol'), table_name='klines')
    op.drop_index(op.f('ix_klines_open_time'), table_name='klines')
    op.drop_table('klines')