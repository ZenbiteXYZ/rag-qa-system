"""Indexes (one Qdrant collection each) and which versions are built in them."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ragqa.domain.jobs import JobStatus
from ragqa.storage.postgres.models.base import Base, created_at, one_of

DISTANCES = ("cosine", "dot", "euclid")


class IndexRecord(Base):
    __tablename__ = "indexes"
    __table_args__ = (
        CheckConstraint("dense_vector_size > 0", name="dense_vector_size_positive"),
        CheckConstraint(one_of("dense_distance", DISTANCES), name="dense_distance_known"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    collection: Mapped[str] = mapped_column(Text, unique=True)
    dense_model: Mapped[str] = mapped_column(Text)
    dense_vector_size: Mapped[int] = mapped_column(Integer)
    dense_distance: Mapped[str] = mapped_column(Text, server_default="cosine")
    sparse_model: Mapped[str | None] = mapped_column(Text)
    chunking_strategy: Mapped[str] = mapped_column(Text)
    chunking_params: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = created_at()


class VersionIndexBuild(Base):
    """Whether a version is built in an index. A version can be in several indexes."""

    __tablename__ = "version_index_builds"
    __table_args__ = (CheckConstraint(one_of("status", JobStatus), name="status_known"),)

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("versions.id", ondelete="CASCADE"), primary_key=True
    )
    index_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("indexes.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(Text)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    built_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
