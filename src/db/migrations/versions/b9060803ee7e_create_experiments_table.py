"""create experiments table

Revision ID: b9060803ee7e
Revises: a8060803ee7d
Create Date: 2026-07-13 08:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b9060803ee7e'
down_revision: Union[str, None] = 'a8060803ee7d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'experiments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_name', sa.String(length=50), nullable=False),
        sa.Column('dataset_version', sa.String(length=20), nullable=False),
        sa.Column('parameters', sa.String(length=500), nullable=False),
        sa.Column('metrics', sa.String(length=500), nullable=False),
        sa.Column('git_sha', sa.String(length=40), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_experiments_model_name'), 'experiments', ['model_name'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_experiments_model_name'), table_name='experiments')
    op.drop_table('experiments')