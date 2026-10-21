"""Chunk store against a real Qdrant (BM25 is computed by the server).

RAGQA_TEST_QDRANT_URL=http://localhost:6333 uv run pytest -m integration
"""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from ragqa.domain.documents import Chunk, DocumentMetadata, EmbeddedChunk
from ragqa.domain.indexes import Index
from ragqa.storage.qdrant.store import ChunkStore, VersionRef

URL = os.environ.get("RAGQA_TEST_QDRANT_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(URL is None, reason="RAGQA_TEST_QDRANT_URL not set"),
]


@pytest.fixture
async def setup() -> AsyncIterator[tuple[ChunkStore, Index, AsyncQdrantClient]]:
    client = AsyncQdrantClient(url=URL)
    name = f"test_{uuid.uuid4().hex[:8]}"
    index = Index(
        id=1,
        collection=f"chunks_{name}",
        name=name,
        dense_model="test",
        dense_vector_size=4,
        chunking_strategy="fixed",
    )
    yield ChunkStore(client), index, client
    await client.delete_collection(index.collection)
    await client.close()


def chunks(n: int, text: str = "kubectl rollout restart deployment") -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk=Chunk(
                chunk_index=i, start=i * 10, end=i * 10 + 5, text=f"{text} {i}", token_count=5
            ),
            dense=(0.1 * (i + 1), 0.2, 0.3, 0.4),
        )
        for i in range(n)
    ]


def ref(document_id: uuid.UUID, version: int) -> VersionRef:
    return VersionRef(
        document_id=document_id,
        version_id=uuid.uuid4(),
        version=version,
        title="Guide",
        metadata=DocumentMetadata(doc_type="task"),
    )


async def test_upsert_flip_and_delete(setup: tuple[ChunkStore, Index, AsyncQdrantClient]) -> None:
    store, index, client = setup
    document_id = uuid.uuid4()
    v1, v2 = ref(document_id, 1), ref(document_id, 2)

    await store.ensure_collection(index)
    await store.ensure_collection(index)  # idempotent
    await store.upsert(index, v1, chunks(3))
    await store.upsert(index, v1, chunks(3))  # a retry overwrites
    await store.upsert(index, v2, chunks(2))

    assert await store.count(index) == 5
    assert await store.count(index, is_latest=True) == 0  # invisible until flipped
    await store.set_latest(index, document_id, v1.version_id)
    assert await store.count(index, is_latest=True) == 3
    await store.set_latest(index, document_id, v2.version_id)
    assert await store.count(index, is_latest=True, version_id=str(v2.version_id)) == 2
    assert await store.count(index, is_latest=True) == 2

    info = await client.get_collection(index.collection)
    assert set(info.payload_schema) >= {"document_id", "version_id", "is_latest", "doc_date"}

    await store.delete_version(index, v1.version_id)
    assert await store.count(index) == 2
    await store.delete_document(index, document_id)
    assert await store.count(index) == 0


async def test_wrong_vector_size_is_rejected(
    setup: tuple[ChunkStore, Index, AsyncQdrantClient],
) -> None:
    store, index, _ = setup
    await store.ensure_collection(index)
    bad = [EmbeddedChunk(chunk=chunks(1)[0].chunk, dense=(0.1, 0.2))]

    with pytest.raises(ValueError, match="expects 4"):
        await store.upsert(index, ref(uuid.uuid4(), 1), bad)
