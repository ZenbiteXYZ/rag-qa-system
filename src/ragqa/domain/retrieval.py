"""What the retrieve stage asks for and what it gets back (ADR-0007, ADR-0018)."""

import uuid
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SearchMode = Literal["dense", "sparse", "hybrid"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MetadataFilter(_Model):
    """Document-level filters (ADR-0007). Set by the user or extracted by Self-Query.

    Empty fields mean "no filter". `version` and `is_latest` are system filters and are not
    here, so Self-Query can never produce them.
    """

    document_ids: tuple[uuid.UUID, ...] = ()  # from a resolved document title
    doc_type: str | None = None
    tags: tuple[str, ...] = ()  # any of
    author: str | None = None
    language: str | None = None
    date_from: date | None = None  # inclusive, on `doc_date`
    date_to: date | None = None


class RetrievalQuery(_Model):
    """Texts per branch are chosen by the caller (ARCHITECTURE section 4): the dense
    branch may get the HyDE text, the sparse branch never does."""

    mode: SearchMode
    dense_text: str | None = None
    sparse_text: str | None = None
    filters: MetadataFilter = MetadataFilter()
    version: int | None = None  # None = latest versions only (`is_latest`)
    branch_limit: int = Field(default=20, gt=0)  # per prefetch branch in hybrid mode
    top_k: int = Field(default=10, gt=0)

    @model_validator(mode="after")
    def _texts_for_mode(self) -> Self:
        if self.mode in ("dense", "hybrid") and not self.dense_text:
            raise ValueError(f"mode {self.mode} needs dense_text")
        if self.mode in ("sparse", "hybrid") and not self.sparse_text:
            raise ValueError(f"mode {self.mode} needs sparse_text")
        return self


class RetrievedChunk(_Model):
    """A chunk as read back from the Qdrant payload, plus its score."""

    point_id: str
    document_id: uuid.UUID
    version_id: uuid.UUID
    version: int
    title: str
    section_path: tuple[str, ...]
    chunk_index: int
    start: int  # span in the version's parsed text, used by evidence coverage
    end: int
    text: str
    has_code: bool
    score: float  # comparable only within one query: RRF scores are not similarities
