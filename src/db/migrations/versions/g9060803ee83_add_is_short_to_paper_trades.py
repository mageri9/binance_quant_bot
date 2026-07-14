"""add is_short to paper trades

Revision ID: g9060803ee83
Revises: f9060803ee82
Create Date: 2026-07-15 01:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g9060803ee83"
down_revision: Union[str, None] = "f9060803ee82"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Добавляем колонку в SQLite через Batch Mode
    with op.batch_alter_table("paper_trades") as batch_op:
        batch_op.add_column(
            sa.Column("is_short", sa.Boolean(), nullable=False, server_default="0")
        )

    # 2. Вычисляем и восстанавливаем is_short для всех старых записей в БД
    op.execute("""
        UPDATE paper_trades
        SET is_short = 1
        WHERE (sl_price > entry_price) OR (tp_price < entry_price)
    """)


def downgrade() -> None:
    with op.batch_alter_table("paper_trades") as batch_op:
        batch_op.drop_column('is_short')