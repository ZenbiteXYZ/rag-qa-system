"""Chunkers and their registry. Implementations are registered in `default_registry`."""

from ragqa.ingestion.chunkers.langchain_chunkers import FixedSizeChunker, RecursiveChunker
from ragqa.ingestion.chunkers.registry import (
    Chunker,
    ChunkerFactory,
    ChunkerRegistry,
    TokenCounter,
    TokenCounterFactory,
    UnknownStrategyError,
)


def default_registry() -> ChunkerRegistry:
    registry = ChunkerRegistry()
    registry.register(FixedSizeChunker.strategy, FixedSizeChunker)
    registry.register(RecursiveChunker.strategy, RecursiveChunker)
    return registry


__all__ = [
    "Chunker",
    "ChunkerFactory",
    "ChunkerRegistry",
    "FixedSizeChunker",
    "RecursiveChunker",
    "TokenCounter",
    "TokenCounterFactory",
    "UnknownStrategyError",
    "default_registry",
]
