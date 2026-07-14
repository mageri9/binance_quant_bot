"""add is_subscribed to user safely

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
    # Безопасная проверка существования таблицы перед изменением
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if 'users' not in tables:
        # Если таблицы users почему-то нет в БД, создаем её с нуля
        op.create_table(
            'users',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('user_id', sa.BigInteger(), nullable=False),
            sa.Column('username', sa.String(length=64), nullable=True),
            sa.Column('full_name', sa.String(length=256), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
            sa.Column('is_blocked', sa.Boolean(), nullable=False, server_default=sa.text('0')),
            sa.Column('is_subscribed', sa.Boolean(), nullable=False, server_default=sa.text('1')),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('user_id')
        )
        op.create_index(op.f('ix_users_user_id'), 'users', ['user_id'], unique=True)
    else:
        # Если таблица users уже есть, просто добавляем колонку, если её там еще нет
        columns = [c['name'] for c in inspector.get_columns('users')]
        if 'is_subscribed' not in columns:
            op.add_column(
                'users',
                sa.Column('is_subscribed', sa.Boolean(), nullable=False, server_default=sa.text('1'))
            )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if 'users' in tables:
        columns = [c['name'] for c in inspector.get_columns('users')]
        if 'is_subscribed' in columns:
            op.drop_column('users', 'is_subscribed')