"""store trade and paper financial values as NUMERIC

Revision ID: l9060803ee88
Revises: k9060803ee87
"""

from alembic import op
import sqlalchemy as sa


revision = "l9060803ee88"
down_revision = "k9060803ee87"
branch_labels = None
depends_on = None


_NUMERIC = sa.Numeric(38, 18)


def upgrade() -> None:
    # batch mode keeps SQLite legacy databases migratable as well as Postgres.
    with op.batch_alter_table("trades") as batch:
        for column in ("entry_price", "exit_price", "amount", "sl_price", "tp_price", "pnl"):
            batch.alter_column(column, existing_type=sa.Float(), type_=_NUMERIC)
    with op.batch_alter_table("portfolios") as batch:
        for column in ("balance", "cash", "positions_value"):
            batch.alter_column(column, existing_type=sa.Float(), type_=_NUMERIC)


def downgrade() -> None:
    with op.batch_alter_table("portfolios") as batch:
        for column in ("balance", "cash", "positions_value"):
            batch.alter_column(column, existing_type=_NUMERIC, type_=sa.Float())
    with op.batch_alter_table("trades") as batch:
        for column in ("entry_price", "exit_price", "amount", "sl_price", "tp_price", "pnl"):
            batch.alter_column(column, existing_type=_NUMERIC, type_=sa.Float())
