"""Conversations and their messages (story C4). Written by the API after each answer."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ragqa.storage.postgres.models.base import Base, created_at, one_of

ROLES = ("system", "user", "assistant", "tool")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "position"),
        CheckConstraint(one_of("role", ROLES), name="role_known"),
        CheckConstraint("position >= 0", name="position_not_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    position: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, server_default="")
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    tool_call_id: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    trace_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()
