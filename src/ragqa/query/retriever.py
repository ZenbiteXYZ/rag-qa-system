"""Retriever contract. The implementation embeds `dense_text` with the index's embedder and
runs one Qdrant query (a prefetch branch per vector, fused with RRF in hybrid mode)."""

from typing import Protocol

from ragqa.domain.indexes import Index
from ragqa.domain.retrieval import RetrievalQuery, RetrievedChunk


class Retriever(Protocol):
    async def search(self, index: Index, query: RetrievalQuery) -> list[RetrievedChunk]:
        """At most `query.top_k` chunks, best first."""
        ...
