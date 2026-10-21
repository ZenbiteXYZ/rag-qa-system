"""Chunk store against a real Qdrant (BM25 is computed by the server).

RAGQA_TEST_QDRANT_URL=http://localhost:6333 uv run pytest -m integration
"""

import os
import uuid
from collections.abc import AsyncIterator
from datetime import date

import pytest
from qdrant_client import AsyncQdrantClient

from ragqa.domain.documents import Chunk, DocumentMetadata, EmbeddedChunk
from ragqa.domain.indexes import Index
from ragqa.domain.retrieval import MetadataFilter, RetrievalQuery
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


# --- Search (ADR-0018) ---


def doc_chunks(texts: list[str]) -> list[EmbeddedChunk]:
    """Chunk i has the dense vector e_i, so a dense query can pick it exactly."""
    out = []
    for i, text in enumerate(texts):
        vector = [0.0, 0.0, 0.0, 0.0]
        vector[i % 4] = 1.0
        out.append(
            EmbeddedChunk(
                chunk=Chunk(
                    chunk_index=i, start=i * 100, end=i * 100 + 50, text=text, token_count=7
                ),
                dense=tuple(vector),
            )
        )
    return out


def vref(document_id: uuid.UUID, version: int, **meta: object) -> VersionRef:
    return VersionRef(
        document_id=document_id,
        version_id=uuid.uuid4(),
        version=version,
        title=f"Doc {version}",
        metadata=DocumentMetadata(**meta),  # type: ignore[arg-type]
    )


async def loaded(
    setup: tuple[ChunkStore, Index, AsyncQdrantClient],
) -> tuple[ChunkStore, Index, VersionRef, VersionRef]:
    store, index, _ = setup
    await store.ensure_collection(index)
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    a = vref(doc_a, 1, author="I. Petrov", tags=("ops", "nodes"), doc_date=date(2026, 3, 1))
    b = vref(doc_b, 1, author="A. Ivanova", tags=("security",), doc_date=date(2026, 9, 1))
    await store.upsert(
        index, a, doc_chunks(["kubectl drain evicts pods", "cordon marks a node", "uncordon it"])
    )
    await store.upsert(index, b, doc_chunks(["rbac roles and bindings", "service accounts"]))
    await store.set_latest(index, doc_a, a.version_id)
    await store.set_latest(index, doc_b, b.version_id)
    return store, index, a, b


async def test_dense_sparse_and_hybrid_modes(
    setup: tuple[ChunkStore, Index, AsyncQdrantClient],
) -> None:
    store, index, a, _ = await loaded(setup)

    dense = await store.search(
        index, RetrievalQuery(mode="dense", dense_text="x", top_k=3), [0.0, 1.0, 0.0, 0.0]
    )
    assert dense[0].text == "cordon marks a node"  # chunk 1 of doc A (the vector e_1)

    sparse = await store.search(
        index, RetrievalQuery(mode="sparse", sparse_text="kubectl drain", top_k=3), None
    )
    assert sparse[0].text == "kubectl drain evicts pods"
    assert sparse[0].document_id == a.document_id and sparse[0].version == 1

    hybrid = await store.search(
        index,
        RetrievalQuery(
            mode="hybrid", dense_text="x", sparse_text="rbac roles", top_k=3, branch_limit=5
        ),
        [1.0, 0.0, 0.0, 0.0],
    )
    texts = [c.text for c in hybrid]
    assert "rbac roles and bindings" in texts and "kubectl drain evicts pods" in texts
    assert hybrid[0].token_count == 7


async def test_filters_and_version_selection(
    setup: tuple[ChunkStore, Index, AsyncQdrantClient],
) -> None:
    store, index, a, b = await loaded(setup)
    # Version 2 of document A exists but is not latest yet.
    a2 = vref(a.document_id, 2, author="I. Petrov")
    await store.upsert(index, a2, doc_chunks(["new drain behaviour"]))

    async def search(filters: MetadataFilter, version: int | None = None) -> set[str]:
        query = RetrievalQuery(
            mode="dense", dense_text="x", filters=filters, version=version, top_k=20
        )
        return {c.title for c in await store.search(index, query, [1.0, 0.0, 0.0, 0.0])}

    assert await search(MetadataFilter()) == {"Doc 1"}  # only latest versions, both documents
    only_a = await store.search(
        index,
        RetrievalQuery(
            mode="dense",
            dense_text="x",
            filters=MetadataFilter(document_ids=(a.document_id,)),
            top_k=20,
        ),
        [1.0, 0.0, 0.0, 0.0],
    )
    assert {c.document_id for c in only_a} == {a.document_id}
    assert {c.version for c in only_a} == {1}  # v2 is not flipped yet

    by_author = await store.search(
        index,
        RetrievalQuery(
            mode="dense",
            dense_text="x",
            filters=MetadataFilter(author="A. Ivanova"),
            top_k=20,
        ),
        [1.0, 0.0, 0.0, 0.0],
    )
    assert {c.document_id for c in by_author} == {b.document_id}
    assert await search(MetadataFilter(tags=("nodes", "other"))) == {"Doc 1"}
    assert await search(MetadataFilter(tags=("nothing",))) == set()
    assert await search(MetadataFilter(date_from=date(2026, 6, 1))) == {"Doc 1"}
    in_march = await store.search(
        index,
        RetrievalQuery(
            mode="dense",
            dense_text="x",
            filters=MetadataFilter(date_to=date(2026, 3, 1)),
            top_k=20,
        ),
        [1.0, 0.0, 0.0, 0.0],
    )
    assert {c.document_id for c in in_march} == {a.document_id}
    # An explicit version ignores is_latest.
    assert await search(MetadataFilter(), version=2) == {"Doc 2"}


async def test_neighbors_by_position(
    setup: tuple[ChunkStore, Index, AsyncQdrantClient],
) -> None:
    store, index, a, _ = await loaded(setup)

    found = await store.get_chunks(index, a.version_id, [0, 2, 9])

    assert [(c.chunk_index, c.is_neighbor) for c in found] == [(0, True), (2, True)]
    assert await store.get_chunks(index, a.version_id, []) == []
