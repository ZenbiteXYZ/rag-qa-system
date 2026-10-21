"""One question, end to end: resolve the request, run the pipeline, store the conversation.

The API handlers and the tests call this; the evaluation runner reaches it over HTTP.
"""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.entities import SpanType
from pydantic import ValidationError

from ragqa.domain.conversation import ChatMessage, NewMessage
from ragqa.domain.indexes import Index
from ragqa.domain.pipeline import PipelineConfig, PipelineState
from ragqa.domain.retrieval import MetadataFilter
from ragqa.llm.types import TextDelta
from ragqa.observability.logging import get_logger
from ragqa.query.runner import QueryDeps, build_stages, run_pipeline
from ragqa.query.stages.generate import GenerateStage
from ragqa.storage.postgres.registry import Registry

log = get_logger(__name__)

_NO_OP_TRACE = "MLFLOW_NO_OP_SPAN_TRACE_ID"


class QueryRequestError(ValueError):
    """The request cannot be run as asked (bad config, unknown index). A client error."""


@dataclass(frozen=True)
class QueryRequest:
    question: str
    index: str | None = None
    filters: MetadataFilter = MetadataFilter()
    version: int | None = None
    conversation_id: uuid.UUID | None = None
    # None: use the stored conversation (or start one). A list, even an empty one, is the
    # whole history and nothing is stored (stateless clients such as the evaluation runner).
    history: tuple[ChatMessage, ...] | None = None
    config: dict[str, Any] | None = None


@dataclass(frozen=True)
class QueryResult:
    state: PipelineState
    conversation_id: uuid.UUID | None
    trace_id: str | None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def assistant_text(state: PipelineState) -> str:
    """What the conversation records as the assistant's turn."""
    if state.answer is not None:
        return state.answer
    if state.clarification is not None:
        c = state.clarification
        options = ", ".join(c.candidates) or "none close enough"
        return f"Which {c.field} do you mean by {c.value!r}? Options: {options}."
    if state.deleted_document is not None:
        d = state.deleted_document
        return f"The document {d.title!r} was deleted on {d.deleted_at:%Y-%m-%d}."
    return "No answer was produced."


class QueryService:
    def __init__(
        self,
        registry: Registry,
        deps: QueryDeps,
        default_config: PipelineConfig,
        *,
        default_index: str | None = None,
    ) -> None:
        self._registry = registry
        self._deps = deps
        self._default_config = default_config
        self._default_index = default_index
        self._stages = build_stages(deps)

    async def _prepare(self, request: QueryRequest) -> tuple[PipelineState, uuid.UUID | None, bool]:
        """The initial state, the conversation to store into, and whether to store."""
        try:
            config = PipelineConfig.model_validate(
                deep_merge(self._default_config.model_dump(), request.config or {})
            )
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise QueryRequestError(f"invalid pipeline config: {problems}") from exc

        index = await self._resolve_index(request.index)
        if config.rerank.enabled and self._deps.reranker is None:
            raise QueryRequestError("reranking is not configured on this deployment")
        if config.retrieve.mode != "dense" and not index.sparse_model:
            raise QueryRequestError(f"index {index.name!r} has no sparse vector")
        if request.conversation_id is not None and request.history is not None:
            raise QueryRequestError("send either conversation_id or history, not both")

        conversation_id = request.conversation_id
        store = request.history is None
        history: tuple[ChatMessage, ...] = request.history or ()
        if store and conversation_id is None:
            conversation_id = await self._registry.create_conversation()
        elif store and conversation_id is not None:
            stored = await self._registry.get_messages(conversation_id)  # NotFoundError: 404
            history = tuple(m.message for m in stored)
        state = PipelineState(
            question=request.question,
            index=index,
            user_filters=request.filters,
            version=request.version,
            history=history,
            config=config,
        )
        return state, conversation_id, store

    async def _resolve_index(self, name: str | None) -> Index:
        wanted = name or self._default_index
        if wanted:
            return await self._registry.get_index(wanted)
        indexes = await self._registry.list_indexes()
        if len(indexes) == 1:
            return indexes[0]
        names = ", ".join(i.name for i in indexes) or "none exist yet"
        raise QueryRequestError(f"name an index with `index` (available: {names})")

    async def _store(
        self, conversation_id: uuid.UUID, state: PipelineState, trace_id: str | None
    ) -> None:
        await self._registry.append_messages(
            conversation_id,
            [
                NewMessage(message=ChatMessage(role="user", content=state.question)),
                NewMessage(
                    message=ChatMessage(role="assistant", content=assistant_text(state)),
                    trace_id=trace_id,
                    citations=tuple(c.model_dump(mode="json") for c in state.citations),
                ),
            ],
        )

    async def answer(self, request: QueryRequest) -> QueryResult:
        state, conversation_id, store = await self._prepare(request)
        with mlflow.start_span("query", SpanType.CHAIN) as span:
            trace_id = span.trace_id if span.trace_id != _NO_OP_TRACE else None
            span.set_inputs({"question": request.question, "index": state.index.name})
            state = await run_pipeline(
                self._stages, state.model_copy(update={"trace_id": trace_id})
            )
            span.set_outputs({"outcome": state.outcome, "answer": state.answer})
        if store and conversation_id is not None:
            await self._store(conversation_id, state, trace_id)
        return QueryResult(state, conversation_id if store else None, trace_id)

    async def answer_stream(self, request: QueryRequest) -> AsyncIterator[dict[str, Any]]:
        """Events for `POST /query/stream`: `stage` per finished stage before generation,
        `token` per text delta, then one `final` with the outcome."""
        state, conversation_id, store = await self._prepare(request)
        with mlflow.start_span("query", SpanType.CHAIN) as span:
            trace_id = span.trace_id if span.trace_id != _NO_OP_TRACE else None
            state = state.model_copy(update={"trace_id": trace_id})
            generate = self._stages[-1]
            assert isinstance(generate, GenerateStage)
            state = await run_pipeline(self._stages[:-1], state)
            for timing in state.timings:
                yield {"event": "stage", "data": {"stage": timing.stage, "ms": timing.ms}}
            if state.outcome is None and generate.enabled(state.config):
                async for item in generate.stream(state):
                    if isinstance(item, TextDelta):
                        yield {"event": "token", "data": {"text": item.text}}
                    else:
                        state = item
            span.set_outputs({"outcome": state.outcome})
        if store and conversation_id is not None:
            await self._store(conversation_id, state, trace_id)
        yield {
            "event": "final",
            "data": QueryResult(state, conversation_id if store else None, trace_id),
        }
