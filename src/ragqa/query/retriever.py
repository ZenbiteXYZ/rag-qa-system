"""Retriever contract and its Qdrant implementation.

The implementation embeds `dense_text` with the index's embedder and runs one Qdrant query (a
prefetch branch per vector, fused with RRF in hybrid mode), ADR-0018.
"""

import uuid
from collections.abc import Sequence
from typing import Protocol

from ragqa.domain.indexes import Index
from ragqa.domain.retrieval import RetrievalQuery, RetrievedChunk
from ragqa.embeddings.tei import Embedder
from ragqa.storage.qdrant.store import ChunkStore


class Retriever(Protocol):
    async def search(self, index: Index, query: RetrievalQuery) -> list[RetrievedChunk]:
        """At most `query.top_k` chunks, best first."""
        ...

    async def neighbors(
        self, index: Index, chunks: Sequence[RetrievedChunk], count: int
    ) -> list[RetrievedChunk]:
        """Up to `count` chunks before and after each given chunk, from the same version,
        without the given chunks themselves. Marked `is_neighbor`."""
        ...


class QdrantRetriever:
    def __init__(self, store: ChunkStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    async def search(self, index: Index, query: RetrievalQuery) -> list[RetrievedChunk]:
        vector = None
        if query.mode in ("dense", "hybrid"):
            vector = (await self._embedder.embed_queries([query.dense_text or ""]))[0]
        return await self._store.search(index, query, vector)

    async def neighbors(
        self, index: Index, chunks: Sequence[RetrievedChunk], count: int
    ) -> list[RetrievedChunk]:
        have = {(c.version_id, c.chunk_index) for c in chunks}
        wanted: dict[uuid.UUID, set[int]] = {}
        for chunk in chunks:
            for offset in range(-count, count + 1):
                position = chunk.chunk_index + offset
                if offset != 0 and position >= 0 and (chunk.version_id, position) not in have:
                    wanted.setdefault(chunk.version_id, set()).add(position)
        found: list[RetrievedChunk] = []
        for version_id, positions in wanted.items():
            found += await self._store.get_chunks(index, version_id, sorted(positions))
        return found
