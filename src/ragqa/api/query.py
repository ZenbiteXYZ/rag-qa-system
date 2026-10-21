"""Question answering: `POST /query` and `POST /query/stream` (stories C1 to C4, D2 to D6)."""

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ragqa.api.deps import ResourcesDep, SettingsDep
from ragqa.domain.conversation import ChatMessage
from ragqa.domain.pipeline import Citation
from ragqa.domain.retrieval import MetadataFilter
from ragqa.query.service import QueryRequest, QueryResult

router = APIRouter(tags=["query"])


class QueryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    index: str | None = None
    filters: MetadataFilter = MetadataFilter()
    version: int | None = Field(default=None, ge=1)
    conversation_id: uuid.UUID | None = None
    # The whole history, oldest first. Stateless: nothing is stored. Exclusive with
    # `conversation_id`.
    history: list[ChatMessage] | None = None
    # Overrides of the default pipeline config, merged field by field (ARCHITECTURE section 4).
    config: dict[str, Any] = Field(default_factory=dict)
    # Return the whole pipeline state. Needs RAGQA_ALLOW_QUERY_DEBUG.
    debug: bool = False


class _Out(BaseModel):
    trace_id: str | None
    conversation_id: uuid.UUID | None
    warnings: list[str] = []
    degraded: list[str] = []
    debug: dict[str, Any] | None = None


class AnswerOut(_Out):
    status: Literal["answer"] = "answer"
    answer: str
    abstained: bool
    citations: list[Citation]


class ClarificationOut(_Out):
    status: Literal["needs_clarification"] = "needs_clarification"
    field: str
    value: str
    candidates: list[str]


class DeletedOut(_Out):
    status: Literal["document_deleted"] = "document_deleted"
    document_title: str
    deleted_at: datetime


class RetrievedOut(_Out):
    """Generation was switched off (`config.generate.enabled=false`), for retrieval-only runs."""

    status: Literal["retrieved"] = "retrieved"
    chunks: int


QueryResponse = Annotated[
    AnswerOut | ClarificationOut | DeletedOut | RetrievedOut, Field(discriminator="status")
]


def to_request(body: QueryBody) -> QueryRequest:
    return QueryRequest(
        question=body.question,
        index=body.index,
        filters=body.filters,
        version=body.version,
        conversation_id=body.conversation_id,
        history=tuple(body.history) if body.history is not None else None,
        config=body.config,
    )


def to_response(
    result: QueryResult, *, debug: bool
) -> AnswerOut | ClarificationOut | DeletedOut | RetrievedOut:
    state = result.state
    common: dict[str, Any] = {
        "trace_id": result.trace_id,
        "conversation_id": result.conversation_id,
        "warnings": list(state.warnings),
        "degraded": list(state.degraded),
        "debug": state.model_dump(mode="json") if debug else None,
    }
    if state.deleted_document is not None:
        return DeletedOut(
            document_title=state.deleted_document.title,
            deleted_at=state.deleted_document.deleted_at,
            **common,
        )
    if state.clarification is not None:
        c = state.clarification
        return ClarificationOut(
            field=c.field, value=c.value, candidates=list(c.candidates), **common
        )
    if state.answer is None:
        return RetrievedOut(chunks=len(state.context_chunks), **common)
    return AnswerOut(
        answer=state.answer,
        abstained=state.abstained,
        citations=list(state.citations),
        **common,
    )


def _check_debug(body: QueryBody, settings: Any) -> None:
    if body.debug and not settings.allow_query_debug:
        raise HTTPException(403, "debug output is disabled on this deployment")


@router.post("/query")
async def query(body: QueryBody, resources: ResourcesDep, settings: SettingsDep) -> QueryResponse:
    """Ask a question. Each outcome is a normal response, discriminated by `status`."""
    _check_debug(body, settings)
    result = await resources.query_service.answer(to_request(body))
    return to_response(result, debug=body.debug)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/query/stream")
async def query_stream(
    body: QueryBody, request: Request, resources: ResourcesDep, settings: SettingsDep
) -> StreamingResponse:
    """The same as `/query` over server-sent events: `stage` events for the stages before
    generation, `token` events with text, then `final` carrying the `/query` response."""
    _check_debug(body, settings)
    events = resources.query_service.answer_stream(to_request(body))
    # Start the generator now, so request errors (unknown index, bad config) are a normal
    # HTTP error and not a broken stream.
    first = await anext(events)

    async def stream() -> AsyncIterator[str]:
        item: dict[str, Any] | None = first
        while item is not None:
            if item["event"] == "final":
                response = to_response(item["data"], debug=body.debug)
                yield _sse("final", response.model_dump(mode="json"))
            else:
                yield _sse(item["event"], item["data"])
            item = await anext(events, None)

    return StreamingResponse(stream(), media_type="text/event-stream")
