"""Response bodies of the document, job and index endpoints."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ragqa.domain.indexes import INDEX_NAME_PATTERN
from ragqa.domain.jobs import JobStatus


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Accepted(_Out):
    document_id: uuid.UUID
    version: int
    version_id: uuid.UUID
    job_id: uuid.UUID
    status: Literal["accepted"] = "accepted"


class NoChanges(_Out):
    status: Literal["no_changes"] = "no_changes"
    document_id: uuid.UUID


class DocumentOut(_Out):
    id: uuid.UUID
    title: str
    latest_version_id: uuid.UUID | None
    created_at: datetime
    deleted_at: datetime | None


class VersionOut(_Out):
    id: uuid.UUID
    document_id: uuid.UUID
    number: int
    filename: str
    source_format: str
    content_hash: str
    size_bytes: int
    metadata: dict[str, Any]
    uploaded_at: datetime


class BuildOut(_Out):
    index_name: str
    status: JobStatus
    chunk_count: int | None
    built_at: datetime | None


class DocumentDetail(DocumentOut):
    latest_version: VersionOut | None
    builds: list[BuildOut]  # of the latest version, one per index


class JobOut(_Out):
    id: uuid.UUID
    version_id: uuid.UUID
    index_id: int
    status: JobStatus
    attempts: int
    error: str | None
    created_at: datetime
    updated_at: datetime


class IndexCreate(BaseModel):
    """`dense_vector_size` is not asked for: the server measures it with the embedder."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=INDEX_NAME_PATTERN)
    dense_model: str = Field(min_length=1)
    dense_distance: Literal["cosine", "dot", "euclid"] = "cosine"
    chunking_strategy: str = Field(min_length=1)
    chunking_params: dict[str, Any] = Field(default_factory=dict)
