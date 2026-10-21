"""Chunkers and their registry. Implementations are registered in `default_registry`."""

from ragqa.ingestion.chunkers.registry import (
    Chunker,
    ChunkerFactory,
    ChunkerRegistry,
    TokenCounter,
    UnknownStrategyError,
)


def default_registry() -> ChunkerRegistry:
    """TODO(author): register the strategies, at least the fixed-size baseline (ADR-0019)."""
    return ChunkerRegistry()


__all__ = [
    "Chunker",
    "ChunkerFactory",
    "ChunkerRegistry",
    "TokenCounter",
    "UnknownStrategyError",
    "default_registry",
]
