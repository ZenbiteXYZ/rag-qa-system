"""Fakes for the query pipeline's collaborators."""

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from ragqa.domain.conversation import ChatMessage
from ragqa.domain.indexes import Index
from ragqa.domain.pipeline import PipelineConfig, PipelineState
from ragqa.domain.retrieval import (
    KnownValues,
    RetrievalQuery,
    RetrievedChunk,
)
from ragqa.llm.types import Completion, StreamEvent, T, TextDelta, ToolSpec, Usage

INDEX = Index(
    id=1,
    name="base",
    collection="chunks_base",
    dense_model="m",
    dense_vector_size=4,
    chunking_strategy="fixed-size",
)


def chunk(
    n: int,
    text: str | None = None,
    *,
    document_id: uuid.UUID | None = None,
    version_id: uuid.UUID | None = None,
    version: int = 1,
    token_count: int | None = 10,
    title: str = "Drain",
    neighbor: bool = False,
) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=f"p{n}-{version_id}",
        document_id=document_id or uuid.UUID(int=1),
        version_id=version_id or uuid.UUID(int=2),
        version=version,
        title=title,
        section_path=("Drain a node", "Options"),
        chunk_index=n,
        start=n * 100,
        end=n * 100 + 50,
        text=text or f"chunk {n} text",
        has_code=False,
        token_count=token_count,
        is_neighbor=neighbor,
        score=1.0 / (n + 1),
    )


def state(question: str = "How do I drain a node?", **update: Any) -> PipelineState:
    config = update.pop("config", PipelineConfig(default_model="haiku-5.5"))
    return PipelineState(question=question, index=INDEX, config=config, **update)


class FakeLLM:
    """Answers from queues: `texts` for `complete`, `structured` (JSON strings) for
    `complete_structured`. Records every call."""

    def __init__(
        self,
        texts: Sequence[str] = ("An answer [1].",),
        structured: Sequence[str] = (),
        fail: Exception | None = None,
    ) -> None:
        self.texts = list(texts)
        self.structured = list(structured)
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def _completion(self, text: str, model: str) -> Completion:
        return Completion(
            text=text,
            model=model,
            resolved_model=f"provider/{model}",
            usage=Usage(prompt_tokens=100, completion_tokens=20),
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        reasoning_effort: str | None = None,
        seed: int | None = None,
    ) -> Completion:
        self.calls.append({"messages": list(messages), "model": model, "effort": reasoning_effort})
        if self.fail is not None:
            raise self.fail
        return self._completion(self.texts.pop(0) if len(self.texts) > 1 else self.texts[0], model)

    async def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        model: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
        seed: int | None = None,
    ) -> tuple[T, Completion]:
        self.calls.append({"messages": list(messages), "model": model, "schema": schema})
        if self.fail is not None:
            raise self.fail
        raw = self.structured.pop(0) if len(self.structured) > 1 else self.structured[0]
        return schema.model_validate_json(raw), self._completion(raw, model)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append({"messages": list(messages), "model": model, "stream": True})
        text = self.texts[0]
        for word in text.split(" "):
            yield TextDelta(text=word + " ")
        yield self._completion(text, model)

    async def models(self) -> list[str]:
        return ["haiku-5.5"]


def self_query_json(**filters: Any) -> str:
    return json.dumps(
        {"semantic_query": filters.pop("semantic_query", "drain"), "filters": filters}
    )


class FakeRetriever:
    def __init__(
        self, chunks: Sequence[RetrievedChunk] = (), neighbors: Sequence[RetrievedChunk] = ()
    ) -> None:
        self.chunks = list(chunks)
        self.neighbor_chunks = list(neighbors)
        self.queries: list[RetrievalQuery] = []

    async def search(self, index: Index, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks[: query.top_k]

    async def neighbors(
        self, index: Index, chunks: Sequence[RetrievedChunk], count: int
    ) -> list[RetrievedChunk]:
        return self.neighbor_chunks


class FakeReranker:
    name = "fake:reranker"

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return self.scores[: len(texts)]


async def no_known_values() -> KnownValues:
    return KnownValues()
