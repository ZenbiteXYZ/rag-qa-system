import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragqa.api.main import create_app
from ragqa.domain.conversation import NewMessage, StoredMessage
from ragqa.domain.indexes import Index
from ragqa.domain.pipeline import PipelineConfig
from ragqa.domain.retrieval import KnownDocument, KnownValues
from ragqa.query.runner import QueryDeps
from ragqa.query.service import QueryService
from ragqa.settings import Settings
from ragqa.storage.postgres.registry import NotFoundError
from tests.conftest import log_buffer
from tests.query_fakes import INDEX, FakeLLM, FakeReranker, FakeRetriever, chunk, self_query_json


class FakeRegistry:
    def __init__(self, indexes: list[Index]) -> None:
        self.indexes = indexes
        self.stored: dict[uuid.UUID, list[StoredMessage]] = {}

    async def get_index(self, name: str) -> Index:
        for index in self.indexes:
            if index.name == name:
                return index
        raise NotFoundError(f"no index named {name!r}")

    async def list_indexes(self) -> list[Index]:
        return self.indexes

    async def create_conversation(self) -> uuid.UUID:
        conversation_id = uuid.uuid4()
        self.stored[conversation_id] = []
        return conversation_id

    async def get_messages(self, conversation_id: uuid.UUID, **_: Any) -> list[StoredMessage]:
        if conversation_id not in self.stored:
            raise NotFoundError(f"no conversation {conversation_id}")
        return self.stored[conversation_id]

    async def append_messages(
        self, conversation_id: uuid.UUID, items: list[NewMessage]
    ) -> list[StoredMessage]:
        rows = self.stored[conversation_id]
        for item in items:
            rows.append(
                StoredMessage(
                    id=uuid.uuid4(),
                    conversation_id=conversation_id,
                    position=len(rows),
                    created_at=datetime.now(UTC),
                    message=item.message,
                    trace_id=item.trace_id,
                    citations=item.citations,
                )
            )
        return rows


class Env:
    def __init__(self, app: FastAPI, client: TestClient, registry: FakeRegistry, llm: FakeLLM):
        self.app, self.client, self.registry, self.llm = app, client, registry, llm


async def _known() -> KnownValues:
    return KnownValues(
        authors=("I. Petrov",),
        documents=(KnownDocument(uuid.UUID(int=8), "Old Guide", datetime(2026, 1, 1, tzinfo=UTC)),),
    )


@pytest.fixture
def env(settings: Settings) -> Iterator[Env]:
    settings = settings.model_copy(update={"allow_query_debug": True})
    app = create_app(settings)
    logs = log_buffer()
    next(logs)
    llm = FakeLLM(texts=["Use drain [1]."])
    registry = FakeRegistry([INDEX])
    deps = QueryDeps(
        llm=llm,
        retriever=FakeRetriever([chunk(0), chunk(1)]),
        known_values=_known,
        token_counter=None,
        reranker=None,
    )
    with TestClient(app) as client:
        app.state.resources.query_service = QueryService(
            registry,  # type: ignore[arg-type]
            deps,
            PipelineConfig(default_model="haiku-5.5"),
        )
        yield Env(app, client, registry, llm)


def ask(env: Env, **body: Any) -> httpx.Response:
    return env.client.post("/query", json={"question": "How do I drain a node?", **body})


def test_answer_with_citations_and_a_new_conversation(env: Env) -> None:
    response = ask(env)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answer" and body["answer"] == "Use drain [1]."
    assert body["abstained"] is False
    assert [c["chunk_index"] for c in body["citations"]] == [0]
    conversation_id = uuid.UUID(body["conversation_id"])
    stored = env.registry.stored[conversation_id]
    assert [m.message.role for m in stored] == ["user", "assistant"]
    assert stored[1].message.content == "Use drain [1]." and len(stored[1].citations) == 1


def test_a_follow_up_in_the_same_conversation_sees_its_history(env: Env) -> None:
    first = ask(env).json()

    ask(env, conversation_id=first["conversation_id"], question="and the flags?")

    sent = env.llm.calls[-1]["messages"]
    assert [m.role for m in sent] == ["system", "user", "assistant", "user"]
    assert "How do I drain a node?" in sent[1].content


def test_inline_history_is_stateless(env: Env) -> None:
    response = ask(
        env, history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )

    assert response.status_code == 200 and response.json()["conversation_id"] is None
    assert env.registry.stored == {}
    assert len(env.llm.calls[-1]["messages"]) == 4


def test_history_and_conversation_id_are_exclusive(env: Env) -> None:
    response = ask(env, conversation_id=str(uuid.uuid4()), history=[])
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_query"


def test_unknown_conversation_is_404(env: Env) -> None:
    assert ask(env, conversation_id=str(uuid.uuid4())).status_code == 404


def test_config_overrides_merge_and_bad_config_is_422(env: Env) -> None:
    ok = ask(env, config={"retrieve": {"mode": "dense", "top_k": 3}, "reasoning_effort": "low"})
    assert ok.status_code == 200 and env.llm.calls[-1]["effort"] == "low"

    unknown = ask(env, config={"retrieve": {"colour": "red"}})
    assert unknown.status_code == 422
    assert "retrieve.colour" in unknown.json()["error"]["message"]

    rerank = ask(env, config={"rerank": {"enabled": True}})
    assert rerank.status_code == 422 and "reranking" in rerank.json()["error"]["message"]

    env.registry.indexes.append(
        INDEX.model_copy(update={"id": 3, "name": "dense", "sparse_model": None})
    )
    no_sparse = ask(env, index="dense", config={"retrieve": {"mode": "hybrid"}})
    assert no_sparse.status_code == 422 and "sparse" in no_sparse.json()["error"]["message"]


def test_unknown_or_ambiguous_index(env: Env) -> None:
    assert ask(env, index="missing").status_code == 404
    env.registry.indexes.append(INDEX.model_copy(update={"id": 2, "name": "other"}))
    response = ask(env)
    assert response.status_code == 422 and "base, other" in response.json()["error"]["message"]


def test_clarification_and_deleted_document_are_normal_responses(env: Env) -> None:
    env.llm.structured = [self_query_json(author="Petro")]
    env.registry.indexes[0] = INDEX
    cfg = {"self_query": {"enabled": True, "match_threshold": 1.0, "candidate_floor": 0.3}}
    asked = ask(env, config=cfg).json()
    assert asked["status"] == "needs_clarification" and asked["field"] == "author"
    assert asked["candidates"] == ["I. Petrov"]
    stored = env.registry.stored[uuid.UUID(asked["conversation_id"])]
    assert "Which author" in stored[1].message.content

    env.llm.structured = [self_query_json(document_title="Old Guide")]
    gone = ask(env, config=cfg).json()
    assert gone["status"] == "document_deleted" and gone["document_title"] == "Old Guide"


def test_retrieval_only_response(env: Env) -> None:
    body = ask(env, config={"generate": {"enabled": False}}).json()
    assert body["status"] == "retrieved" and body["chunks"] == 2


def test_debug_output_is_gated_and_complete(env: Env) -> None:
    body = ask(env, debug=True).json()
    debug = body["debug"]
    assert [t["stage"] for t in debug["timings"]] == ["retrieve", "context", "generate"]
    assert debug["prompt"].startswith("Excerpts:") and len(debug["candidates"]) == 2
    assert debug["llm_calls"][0]["prompt_tokens"] == 100

    env.app.state.settings = env.app.state.settings.model_copy(update={"allow_query_debug": False})
    assert ask(env, debug=True).status_code == 403
    assert ask(env).json()["debug"] is None


def test_stream_events(env: Env) -> None:
    with env.client.stream(
        "POST", "/query/stream", json={"question": "How do I drain a node?"}
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())

    events = [
        (block.split("\n")[0].removeprefix("event: "), json.loads(block.split("\n")[1][6:]))
        for block in text.strip().split("\n\n")
    ]
    names = [name for name, _ in events]
    assert names[:2] == ["stage", "stage"] and names[-1] == "final"
    assert "".join(d["text"] for n, d in events if n == "token").strip() == "Use drain [1]."
    final = events[-1][1]
    assert final["status"] == "answer" and final["citations"][0]["chunk_index"] == 0


def test_stream_request_errors_are_plain_http_errors(env: Env) -> None:
    response = env.client.post("/query/stream", json={"question": "q", "index": "missing"})
    assert response.status_code == 404


def test_request_validation(env: Env) -> None:
    assert env.client.post("/query", json={"question": ""}).status_code == 422
    assert env.client.post("/query", json={"question": "q", "version": 0}).status_code == 422
    assert env.client.post("/query", json={"question": "q", "extra": 1}).status_code == 422


def test_reranker_is_used_when_configured(env: Env) -> None:
    service = env.app.state.resources.query_service
    deps = QueryDeps(
        llm=env.llm,
        retriever=FakeRetriever([chunk(0), chunk(1)]),
        known_values=_known,
        token_counter=None,
        reranker=FakeReranker([0.1, 0.9]),
    )
    env.app.state.resources.query_service = QueryService(
        env.registry,  # type: ignore[arg-type]
        deps,
        PipelineConfig(default_model="haiku-5.5"),
    )
    del service

    body = ask(env, config={"rerank": {"enabled": True, "top_n": 1}}, debug=True).json()

    assert [c["chunk_index"] for c in body["debug"]["ranked"]] == [1]
