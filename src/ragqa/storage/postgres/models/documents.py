"""Documents and their versions."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ragqa.storage.postgres.models.base import Base, created_at


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    # Source of truth for "latest" (ADR-0014). Circular with versions.document_id, hence use_alter.
    latest_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("versions.id", use_alter=True, ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = created_at()
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # tombstone


class Version(Base):
    __tablename__ = "versions"
    __table_args__ = (
        UniqueConstraint("document_id", "number"),
        CheckConstraint("number >= 1", name="number_positive"),
        CheckConstraint("size_bytes >= 0", name="size_not_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    # As uploaded; the parser is chosen by its extension.
    filename: Mapped[str] = mapped_column(Text)
    source_format: Mapped[str] = mapped_column(Text)  # parser registry key, for example "md"
    content_hash: Mapped[str] = mapped_column(Text)  # SHA-256 hex of the raw bytes
    raw_path: Mapped[str] = mapped_column(Text)  # relative to the file store root
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    # Metadata given at upload (story A4). Merged over what the parser extracts.
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb")
    )
    uploaded_at: Mapped[datetime] = created_at()
