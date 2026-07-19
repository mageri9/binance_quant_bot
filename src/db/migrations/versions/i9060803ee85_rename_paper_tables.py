"""rename paper tables to portfolios and trades

Revision ID: i9060803ee85
Revises: h9060803ee84
Create Date: 2026-07-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'i9060803ee85'
down_revision: Union[str, None] = 'h9060803ee84'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Безопасное переименование таблиц в SQLite и Postgres
    op.rename_table('paper_portfolios', 'portfolios')
    op.rename_table('paper_trades', 'trades')


def downgrade() -> None:
    op.rename_table('portfolios', 'paper_portfolios')
    op.rename_table('trades', 'paper_trades')