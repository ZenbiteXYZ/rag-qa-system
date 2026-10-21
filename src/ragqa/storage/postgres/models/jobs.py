"""Ingestion jobs (ADR-0011). Their status lives here, not in the queue message."""

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ragqa.domain.jobs import JobStatus
from ragqa.storage.postgres.models.base import Base, created_at, one_of


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(one_of("status", JobStatus), name="status_known"),
        CheckConstraint("attempts >= 0", name="attempts_not_negative"),
        # The republisher looks for jobs stuck in `queued`.
        Index(None, "status", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("versions.id", ondelete="CASCADE"), index=True
    )
    index_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("indexes.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(Text, server_default=JobStatus.QUEUED.value)
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
