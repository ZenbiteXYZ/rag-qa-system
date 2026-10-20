"""Ingestion job message, published by the API and consumed by the worker (ADR-0011).

Fields may change with the Postgres registry in slice 2. Keep the message small: the
worker loads everything else from Postgres by id.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class IngestJob(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: UUID
    document_id: UUID
    version: int = Field(ge=1)
    index: str = Field(min_length=1)
