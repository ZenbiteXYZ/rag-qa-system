"""Conversations and messages.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-21
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamp(name: str) -> sa.Column[datetime]:
    return sa.Column(
        name, sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.UUID(), nullable=False),
        _timestamp("created_at"),
        _timestamp("updated_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), server_default="", nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("tool_call_id", sa.Text(), nullable=True),
        sa.Column("citations", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("trace_id", sa.Text(), nullable=True),
        _timestamp("created_at"),
        sa.CheckConstraint(
            "role IN ('system', 'user', 'assistant', 'tool')",
            name=op.f("ck_messages_role_known"),
        ),
        sa.CheckConstraint("position >= 0", name=op.f("ck_messages_position_not_negative")),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_messages_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint(
            "conversation_id", "position", name=op.f("uq_messages_conversation_id_position")
        ),
    )


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("conversations")
