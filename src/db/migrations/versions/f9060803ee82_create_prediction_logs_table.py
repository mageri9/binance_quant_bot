"""create prediction logs table

Revision ID: f9060803ee82
Revises: e9060803ee81
Create Date: 2026-07-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f9060803ee82'
down_revision: Union[str, None] = 'e9060803ee81'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'prediction_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('model_id', sa.String(length=100), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('prediction', sa.Integer(), nullable=False),
        sa.Column('prob_short', sa.Float(), nullable=False),
        sa.Column('prob_hold', sa.Float(), nullable=False),
        sa.Column('prob_long', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_prediction_logs_symbol'), 'prediction_logs', ['symbol'], unique=False)
    op.create_index(op.f('ix_prediction_logs_timestamp'), 'prediction_logs', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_prediction_logs_timestamp'), table_name='prediction_logs')
    op.drop_index(op.f('ix_prediction_logs_symbol'), table_name='prediction_logs')
    op.drop_table('prediction_logs')