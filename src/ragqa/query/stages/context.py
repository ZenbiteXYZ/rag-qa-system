"""Context assembly: the chunks that go into the prompt, within a token budget, optionally
with their neighbors in the document."""

import asyncio
import math

from ragqa.domain.indexes import Index
from ragqa.domain.pipeline import PipelineConfig, PipelineState
from ragqa.domain.retrieval import RetrievedChunk
from ragqa.ingestion.chunkers import TokenCounterFactory
from ragqa.query.retriever import Retriever


def order_with_neighbors(
    source: tuple[RetrievedChunk, ...], extra: list[RetrievedChunk], count: int
) -> list[RetrievedChunk]:
    """Rank order is kept; each chunk comes with its neighbors, in document order."""
    seen: set[tuple[object, int]] = set()
    ordered: list[RetrievedChunk] = []
    for chunk in source:
        near = [
            n
            for n in extra
            if n.version_id == chunk.version_id
            and 0 < abs(n.chunk_index - chunk.chunk_index) <= count
        ]
        group = sorted([chunk, *near], key=lambda c: c.chunk_index)
        for item in group:
            key = (item.version_id, item.chunk_index)
            if key not in seen:
                seen.add(key)
                ordered.append(item)
    return ordered


class ContextStage:
    name = "context"
    reads = ("ranked", "candidates", "index")
    writes = ("context_chunks",)
    optional = False

    def __init__(self, retriever: Retriever, token_counter: TokenCounterFactory | None) -> None:
        self._retriever = retriever
        self._token_counter = token_counter

    def enabled(self, config: PipelineConfig) -> bool:
        return config.retrieve.enabled

    async def run(self, state: PipelineState) -> PipelineState:
        config = state.config.context
        source = state.ranked or state.candidates
        if not source:
            return state
        extra: list[RetrievedChunk] = []
        if config.neighbors:
            extra = await self._retriever.neighbors(state.index, source, config.neighbors)
        ordered = order_with_neighbors(source, extra, config.neighbors)
        counts, estimated = await self._count(state.index, ordered)

        selected: list[RetrievedChunk] = []
        total = 0
        for chunk, tokens in zip(ordered, counts, strict=True):
            if selected and total + tokens > config.max_tokens:
                break
            selected.append(chunk)
            total += tokens
        update: dict[str, object] = {"context_chunks": tuple(selected)}
        if estimated:
            update["warnings"] = (*state.warnings, "context token counts are estimated")
        return state.model_copy(update=update)

    async def _count(self, index: Index, chunks: list[RetrievedChunk]) -> tuple[list[int], bool]:
        missing = [c for c in chunks if c.token_count is None]
        by_point: dict[str, int] = {}
        estimated = False
        if missing and self._token_counter is not None:
            counter = await asyncio.to_thread(self._token_counter, index)
            counts = await asyncio.to_thread(lambda: [counter.count(c.text) for c in missing])
            by_point = {c.point_id: n for c, n in zip(missing, counts, strict=True)}
        elif missing:
            estimated = True
            by_point = {c.point_id: math.ceil(len(c.text) / 4) for c in missing}
        return [c.token_count or by_point.get(c.point_id, 0) for c in chunks], estimated
