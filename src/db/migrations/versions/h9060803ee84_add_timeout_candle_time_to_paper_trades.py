"""add timeout_candle_time to paper trades

Revision ID: h9060803ee84
Revises: g9060803ee83
Create Date: 2026-07-16 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'h9060803ee84'
down_revision: Union[str, None] = 'g9060803ee83'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('paper_trades') as batch_op:
        batch_op.add_column(
            sa.Column('timeout_candle_time', sa.BigInteger(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table('paper_trades') as batch_op:
        batch_op.drop_column('timeout_candle_time')