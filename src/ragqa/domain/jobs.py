"""Ingestion job message, published by the API and consumed by the worker (ADR-0011).

Keep the message small: the worker loads everything else from Postgres by id. The job's
status lives in Postgres (`jobs.status`), not in the message.
"""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"  # permanent error, no retries
    DEAD = "dead"  # retries exhausted, the message is in the dead-letter queue


class IngestJob(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: UUID
    document_id: UUID
    version_id: UUID
    version: int = Field(ge=1)  # the version number, for logs
    index_id: int = Field(ge=1)
