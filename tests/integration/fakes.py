"""Test doubles for the parts that need a model or a user-written implementation."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from ragqa.domain.documents import Chunk, ParsedDocument
from ragqa.ingestion.chunkers import ChunkerRegistry, TokenCounter, TokenCounterFactory

DIMENSIONS = 4


class FakeEmbedder:
    """Deterministic vectors from a hash of the text, so equal texts embed equally."""

    def __init__(self) -> None:
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(b + 1) / 256 for b in digest[:DIMENSIONS]]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class ParagraphChunker:
    """One chunk per blank-line separated block. Stands in for the real chunkers."""

    strategy = "paragraphs"

    def __init__(self, params: Mapping[str, Any], counter: TokenCounter) -> None:
        self._counter = counter

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        position = 0
        for block in doc.text.split("\n\n"):
            start = doc.text.index(block, position) if block else position
            position = start + len(block)
            if block.strip():
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        start=start,
                        end=position,
                        text=block,
                        section_path=doc.section_path_at(start),
                        token_count=self._counter.count(block),
                    )
                )
        return chunks


def make_chunkers() -> ChunkerRegistry:
    registry = ChunkerRegistry()
    registry.register("paragraphs", ParagraphChunker)
    return registry


def word_counter_factory() -> TokenCounterFactory:
    return lambda index: WordCounter()
