"""Chunks in Qdrant: one collection per index, with a dense and a BM25 vector (ADR-0013, ADR-0018).

`is_latest` is a copy of Postgres' latest pointer for fast filtering (ADR-0014). New points
are written with `is_latest=false` and flipped by `set_latest` after Postgres commits.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from ragqa.domain.documents import DocumentMetadata, EmbeddedChunk
from ragqa.domain.indexes import Index

DENSE = "dense"
SPARSE = "bm25"
# Fixed namespace, so a (version, index, chunk) always maps to the same point id.
POINT_NAMESPACE = uuid.UUID("6f1c3b0e-6d8a-4c55-9b3e-2a7d1f4e8c90")
_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}
# Filtered fields need a payload index, or filtered search slows down.
PAYLOAD_INDEXES = {
    "document_id": models.PayloadSchemaType.KEYWORD,
    "version_id": models.PayloadSchemaType.KEYWORD,
    "version": models.PayloadSchemaType.INTEGER,
    "is_latest": models.PayloadSchemaType.BOOL,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "tags": models.PayloadSchemaType.KEYWORD,
    "author": models.PayloadSchemaType.KEYWORD,
    "language": models.PayloadSchemaType.KEYWORD,
    "doc_date": models.PayloadSchemaType.DATETIME,
}


@dataclass(frozen=True)
class VersionRef:
    """What every chunk of one version shares in its payload."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    version: int
    title: str
    metadata: DocumentMetadata


def point_id(version_id: uuid.UUID, index_id: int, chunk_index: int) -> str:
    """Deterministic, so a retried upsert overwrites instead of duplicating."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{version_id}/{index_id}/{chunk_index}"))


def chunk_payload(ref: VersionRef, embedded: EmbeddedChunk) -> dict[str, Any]:
    chunk = embedded.chunk
    meta = ref.metadata
    return {
        "document_id": str(ref.document_id),
        "version_id": str(ref.version_id),
        "version": ref.version,
        "is_latest": False,
        "title": ref.title,
        "doc_type": meta.doc_type,
        "tags": list(meta.tags),
        "author": meta.author,
        "language": meta.language,
        # RFC 3339, which the datetime payload index accepts.
        "doc_date": datetime.combine(meta.doc_date, time()).isoformat() + "Z"
        if meta.doc_date
        else None,
        "section_path": list(chunk.section_path),
        "chunk_index": chunk.chunk_index,
        "start": chunk.start,
        "end": chunk.end,
        "has_code": chunk.has_code,
        "text": chunk.text,
    }


def _match(**conditions: Any) -> list[models.Condition]:
    return [
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in conditions.items()
    ]


class ChunkStore:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def ensure_collection(self, index: Index) -> None:
        """Create the collection with both vectors and the payload indexes. Idempotent.

        Both vectors must exist from the start: a sparse vector cannot be added to an
        existing collection (ADR-0018 spike).
        """
        if not await self._client.collection_exists(index.collection):
            await self._client.create_collection(
                collection_name=index.collection,
                vectors_config={
                    DENSE: models.VectorParams(
                        size=index.dense_vector_size, distance=_DISTANCES[index.dense_distance]
                    )
                },
                sparse_vectors_config=(
                    {SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                    if index.sparse_model
                    else None
                ),
            )
        for field, schema in PAYLOAD_INDEXES.items():
            await self._client.create_payload_index(index.collection, field, field_schema=schema)

    async def upsert(
        self,
        index: Index,
        ref: VersionRef,
        chunks: Sequence[EmbeddedChunk],
        *,
        batch_size: int = 64,
    ) -> int:
        """ADR-0014 step 1: write the version's chunks, invisible to default queries."""
        points = []
        for embedded in chunks:
            if len(embedded.dense) != index.dense_vector_size:
                size = index.dense_vector_size
                raise ValueError(f"vector has {len(embedded.dense)} values, index expects {size}")
            vector: dict[str, Any] = {DENSE: list(embedded.dense)}
            if index.sparse_model:
                # Qdrant computes BM25 from the raw text on the server (ADR-0018).
                vector[SPARSE] = models.Document(text=embedded.chunk.text, model=index.sparse_model)
            points.append(
                models.PointStruct(
                    id=point_id(ref.version_id, index.id, embedded.chunk.chunk_index),
                    vector=vector,
                    payload=chunk_payload(ref, embedded),
                )
            )
        for start in range(0, len(points), batch_size):
            await self._client.upsert(index.collection, points[start : start + batch_size])
        return len(points)

    async def set_latest(self, index: Index, document_id: uuid.UUID, version_id: uuid.UUID) -> None:
        """ADR-0014 step 3: the new version first, then the old ones, so there is never a gap."""
        await self._client.set_payload(
            index.collection,
            payload={"is_latest": True},
            points=models.Filter(must=_match(version_id=str(version_id))),
        )
        await self._client.set_payload(
            index.collection,
            payload={"is_latest": False},
            points=models.Filter(
                must=_match(document_id=str(document_id), is_latest=True),
                must_not=_match(version_id=str(version_id)),
            ),
        )

    async def delete_version(self, index: Index, version_id: uuid.UUID) -> None:
        await self._client.delete(
            index.collection, points_selector=models.Filter(must=_match(version_id=str(version_id)))
        )

    async def delete_document(self, index: Index, document_id: uuid.UUID) -> None:
        await self._client.delete(
            index.collection,
            points_selector=models.Filter(must=_match(document_id=str(document_id))),
        )

    async def count(self, index: Index, **conditions: Any) -> int:
        result = await self._client.count(
            index.collection,
            count_filter=models.Filter(must=_match(**conditions)) if conditions else None,
            exact=True,
        )
        return result.count
