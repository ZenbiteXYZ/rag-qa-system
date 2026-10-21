"""Chunker contract and registry: `indexes.chunking_strategy` -> chunker (ADR-0013).

A chunker may wrap a library splitter (LangChain, LlamaIndex). Whatever it uses inside,
its output must satisfy `check_chunks`: spans are offsets into `ParsedDocument.text`,
because evidence coverage is measured on those spans (ADR-0019, ADR-0021).
"""

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from ragqa.domain.documents import Chunk, ParsedDocument


class TokenCounter(Protocol):
    """Counts tokens with the index's embedder tokenizer, so `Chunk.token_count` and the
    chunk size limit mean the same thing as the model's context window."""

    def count(self, text: str) -> int: ...


class Chunker(Protocol):
    strategy: str  # the key stored in `indexes.chunking_strategy`

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Synchronous and CPU-bound: the worker runs it in an executor (ADR-0011)."""
        ...


# Builds a chunker from `indexes.chunking_params`. Validate the params here (for example
# with a pydantic model) and raise `ValueError` for bad ones, so `POST /indexes` can
# reject them before any document is ingested.
ChunkerFactory = Callable[[Mapping[str, Any], TokenCounter], Chunker]


class UnknownStrategyError(ValueError):
    pass


class ChunkerRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ChunkerFactory] = {}

    def register(self, strategy: str, factory: ChunkerFactory) -> None:
        if strategy in self._factories:
            raise ValueError(f"chunking strategy {strategy!r} is already registered")
        self._factories[strategy] = factory

    def build(self, strategy: str, params: Mapping[str, Any], counter: TokenCounter) -> Chunker:
        try:
            factory = self._factories[strategy]
        except KeyError:
            known = ", ".join(self.strategies) or "none"
            raise UnknownStrategyError(
                f"unknown chunking strategy {strategy!r}; known: {known}"
            ) from None
        return factory(params, counter)

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
