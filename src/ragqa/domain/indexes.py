"""Index configuration: one Qdrant collection per (dense model, chunking strategy) (ADR-0013)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

INDEX_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,62}$"


def collection_for(index_name: str) -> str:
    """Qdrant collection name for a new index. Stored separately, so a rename keeps it."""
    return f"chunks_{index_name}"


class IndexConfig(BaseModel):
    """What a client sends to create an index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=INDEX_NAME_PATTERN)
    dense_model: str = Field(min_length=1)
    dense_vector_size: int = Field(gt=0)
    dense_distance: Literal["cosine", "dot", "euclid"] = "cosine"
    sparse_model: str | None = "Qdrant/bm25"
    chunking_strategy: str = Field(min_length=1)
    chunking_params: dict[str, Any] = Field(default_factory=dict)


class Index(IndexConfig):
    """An index as stored in the registry."""

    id: int
    collection: str
