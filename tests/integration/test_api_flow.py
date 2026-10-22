"""The API against a real Postgres and Qdrant, with a fake embedder and no broker.

Publishing fails (nothing listens on the broker URL), which is the case the republisher
exists for: the upload still succeeds and the job stays `queued`.
"""

import json
import uuid
from datetime import timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from ragqa.domain.conversation import ChatMessage, NewMessage
from ragqa.domain.indexes import IndexConfig
from ragqa.domain.pipeline import PipelineConfig
from ragqa.ingestion.service import IngestionService
from ragqa.query.retriever import QdrantRetriever
from ragqa.query.runner import QueryDeps
from ragqa.query.service import QueryService
from ragqa.settings import Settings
from tests.integration.conftest import POSTGRES_DSN, QDRANT_URL
from tests.integration.fakes import DIMENSIONS
from tests.query_fakes import FakeLLM, self_query_json

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        POSTGRES_DSN is None or QDRANT_URL is None,
        reason="RAGQA_TEST_POSTGRES_DSN and RAGQA_TEST_QDRANT_URL are not both set",
    ),
]

MODEL = Settings.model_fields["embedder_model"].default
GUIDE = b"# Drain\n\nUse kubectl drain to empty a node.\n\nIt evicts the pods.\n"


def _service(app: FastAPI) -> IngestionService:
    backends = app.state.resources.backends
    return IngestionService(
        registry=backends.registry,
        files=backends.files,
        chunk_store=backends.chunk_store,
        embedder=backends.embedder,
        parsers=backends.parsers,
        chunkers=backends.chunkers,
        token_counter=backends.token_counter,
    )


async def _create_index(client: httpx.AsyncClient, name: str = "base") -> dict[str, Any]:
    response = await client.post(
        "/indexes",
        json={"name": name, "dense_model": MODEL, "chunking_strategy": "paragraphs"},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_index_creation_measures_the_vector_size_and_validates(
    client: httpx.AsyncClient,
) -> None:
    index = await _create_index(client)
    assert index["dense_vector_size"] == DIMENSIONS
    assert index["collection"] == "chunks_base"
    assert [i["name"] for i in (await client.get("/indexes")).json()] == ["base"]

    duplicate = await client.post(
        "/indexes",
        json={"name": "base", "dense_model": MODEL, "chunking_strategy": "paragraphs"},
    )
    assert duplicate.status_code == 409
    other_model = await client.post(
        "/indexes",
        json={"name": "x", "dense_model": "other/model", "chunking_strategy": "paragraphs"},
    )
    assert other_model.status_code == 422
    unknown_strategy = await client.post(
        "/indexes",
        json={"name": "y", "dense_model": MODEL, "chunking_strategy": "semantic"},
    )
    assert unknown_strategy.status_code == 422
    assert "paragraphs" in unknown_strategy.json()["error"]["message"]


async def test_upload_ingest_read_and_delete(app: FastAPI, client: httpx.AsyncClient) -> None:
    await _create_index(client)

    # The only index is used when the upload names none.
    response = await client.post(
        "/documents",
        files={"file": ("drain.md", GUIDE)},
        data={"title": "Drain a node", "metadata": '{"author": "I. Petrov", "tags": ["ops"]}'},
    )
    assert response.status_code == 202, response.text
    accepted = response.json()
    document_id, job_id = accepted["document_id"], accepted["job_id"]
    assert accepted["version"] == 1

    job = (await client.get(f"/jobs/{job_id}")).json()
    assert job["status"] == "queued"  # publishing failed, so it waits for the republisher

    registry = app.state.resources.backends.registry
    stuck = await registry.stuck_jobs(timedelta(0))
    assert [j.job_id for j in stuck] == [uuid.UUID(job_id)]
    await _service(app).ingest(stuck[0])

    assert (await client.get(f"/jobs/{job_id}")).json()["status"] == "ready"
    detail = (await client.get(f"/documents/{document_id}")).json()
    assert detail["title"] == "Drain a node"
    assert detail["latest_version"]["number"] == 1
    assert detail["latest_version"]["metadata"] == {"author": "I. Petrov", "tags": ["ops"]}
    assert "raw_path" not in detail["latest_version"]
    assert [(b["index_name"], b["status"], b["chunk_count"]) for b in detail["builds"]] == [
        ("base", "ready", 3)
    ]

    same = await client.post(
        f"/documents/{document_id}/versions", files={"file": ("drain.md", GUIDE)}
    )
    assert same.status_code == 200
    assert same.json()["status"] == "no_changes"
    changed = await client.post(
        f"/documents/{document_id}/versions", files={"file": ("drain.md", GUIDE + b"\nMore.\n")}
    )
    assert changed.status_code == 202
    assert changed.json()["version"] == 2
    versions = (await client.get(f"/documents/{document_id}/versions")).json()
    assert [v["number"] for v in versions] == [1, 2]

    deleted = await client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 202
    # The background purge has run by now; the tombstone stays.
    assert (await client.get("/documents")).json() == []
    tombstones = (await client.get("/documents", params={"include_deleted": True})).json()
    assert [d["id"] for d in tombstones] == [document_id]
    assert (await client.get(f"/documents/{document_id}/versions")).json() == []
    qdrant = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    index = await registry.get_index("base")
    assert (await qdrant.count(index.collection)).count == 0
    await qdrant.close()


async def test_upload_needs_an_index_name_when_there_are_several(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    registry = app.state.resources.backends.registry
    for name in ("one", "two"):
        await registry.create_index(
            IndexConfig(
                name=name, dense_model="m", dense_vector_size=4, chunking_strategy="paragraphs"
            )
        )

    response = await client.post("/documents", files={"file": ("drain.md", GUIDE)})
    assert response.status_code == 422
    assert "one, two" in response.json()["error"]["message"]

    unknown = await client.post(
        "/documents", files={"file": ("drain.md", GUIDE)}, data={"index": "missing"}
    )
    assert unknown.status_code == 404


async def test_unknown_ids_are_404(client: httpx.AsyncClient) -> None:
    missing = uuid.uuid4()
    for path in (
        f"/documents/{missing}",
        f"/documents/{missing}/versions",
        f"/jobs/{missing}",
        f"/conversations/{missing}/messages",
    ):
        assert (await client.get(path)).status_code == 404, path
    assert (await client.delete(f"/documents/{missing}")).status_code == 404


async def test_conversation_messages_are_readable(app: FastAPI, client: httpx.AsyncClient) -> None:
    registry = app.state.resources.backends.registry
    conversation_id = await registry.create_conversation()
    await registry.append_messages(
        conversation_id,
        [
            NewMessage(message=ChatMessage(role="user", content="Who wrote it?")),
            NewMessage(
                message=ChatMessage(role="assistant", content="I. Petrov"),
                trace_id="tr-1",
                citations=({"title": "Drain"},),
            ),
        ],
    )

    body = (await client.get(f"/conversations/{conversation_id}/messages")).json()

    assert [m["message"]["role"] for m in body] == ["user", "assistant"]
    assert body[1]["trace_id"] == "tr-1"
    assert body[1]["citations"] == [{"title": "Drain"}]


# --- Question answering over really ingested documents ---

DRAIN = b"# Drain\n\nUse kubectl drain to empty a node.\n\nIt evicts the pods.\n"
RBAC = b"# RBAC\n\nRoles grant permissions to subjects.\n\nBindings attach roles.\n"


async def _upload(
    client: httpx.AsyncClient, name: str, data: bytes, title: str, author: str
) -> dict[str, Any]:
    response = await client.post(
        "/documents",
        files={"file": (name, data)},
        data={"title": title, "metadata": json.dumps({"author": author, "tags": ["ops"]})},
    )
    assert response.status_code == 202, response.text
    body: dict[str, Any] = response.json()
    return body


async def _ingest_all(app: FastAPI) -> None:
    registry = app.state.resources.backends.registry
    for job in await registry.stuck_jobs(timedelta(0)):
        await _service(app).ingest(job)


def _qa(app: FastAPI, llm: FakeLLM) -> None:
    backends = app.state.resources.backends
    deps = QueryDeps(
        llm=llm,
        retriever=QdrantRetriever(backends.chunk_store, backends.embedder),
        known_values=backends.registry.known_filter_values,
        token_counter=backends.token_counter,
    )
    app.state.resources.query_service = QueryService(
        backends.registry, deps, PipelineConfig(default_model="haiku-5.5")
    )


async def _ask(client: httpx.AsyncClient, **body: Any) -> dict[str, Any]:
    response = await client.post(
        "/query",
        json={"question": "Use kubectl drain to empty a node.", "debug": True, **body},
    )
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


async def test_question_answering_over_ingested_documents(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    await _create_index(client)
    drain = await _upload(client, "drain.md", DRAIN, "Drain a node", "I. Petrov")
    rbac = await _upload(client, "rbac.md", RBAC, "RBAC", "A. Ivanova")
    await _ingest_all(app)
    llm = FakeLLM(texts=["Use kubectl drain [1]."])
    _qa(app, llm)

    # Dense: the query text equals a chunk, so the fake embedder finds it exactly.
    body = await _ask(client)
    assert body["status"] == "answer" and body["citations"][0]["title"] == "Drain a node"
    assert body["citations"][0]["document_id"] == drain["document_id"]
    assert body["debug"]["candidates"][0]["text"] == "Use kubectl drain to empty a node."
    assert [t["stage"] for t in body["debug"]["timings"]] == ["retrieve", "context", "generate"]

    # Hybrid: the BM25 branch finds a chunk by its words.
    hybrid = await _ask(
        client,
        question="roles permissions subjects",
        config={"retrieve": {"mode": "hybrid", "top_k": 2}},
    )
    assert {c["title"] for c in hybrid["debug"]["candidates"]} >= {"RBAC"}

    # The conversation was stored with the citation, and can be read back.
    messages = (await client.get(f"/conversations/{body['conversation_id']}/messages")).json()
    assert [m["message"]["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["citations"][0]["title"] == "Drain a node"

    # Self-Query against the real known values: "Petrov" becomes "I. Petrov", which limits the
    # search to the drain document.
    llm.structured = [self_query_json(author="Petrov", semantic_query="rbac roles")]
    filtered = await _ask(
        client, config={"self_query": {"enabled": True}}, question="rbac roles by Petrov"
    )
    assert filtered["debug"]["resolved_filters"]["author"] == "I. Petrov"
    assert {c["document_id"] for c in filtered["debug"]["candidates"]} == {drain["document_id"]}
    assert rbac["document_id"] not in {c["document_id"] for c in filtered["debug"]["candidates"]}

    # A deleted document is reported, not searched.
    assert (await client.delete(f"/documents/{rbac['document_id']}")).status_code == 202
    llm.structured = [self_query_json(document_title="RBAC")]
    gone = await _ask(client, config={"self_query": {"enabled": True}}, question="about RBAC")
    assert gone["status"] == "document_deleted" and gone["document_title"] == "RBAC"


async def test_version_selection_in_queries(app: FastAPI, client: httpx.AsyncClient) -> None:
    await _create_index(client)
    first = await _upload(client, "drain.md", DRAIN, "Drain a node", "I. Petrov")
    await _ingest_all(app)
    changed = DRAIN.replace(b"evicts the pods", b"evicts every pod gracefully")
    response = await client.post(
        f"/documents/{first['document_id']}/versions", files={"file": ("drain.md", changed)}
    )
    assert response.status_code == 202
    await _ingest_all(app)
    _qa(app, FakeLLM(texts=["ok [1]"]))

    latest = await _ask(client, question="It evicts every pod gracefully")
    pinned = await _ask(client, question="It evicts the pods.", version=1)

    assert {c["version"] for c in latest["debug"]["candidates"]} == {2}
    assert {c["version"] for c in pinned["debug"]["candidates"]} == {1}
    assert pinned["debug"]["candidates"][0]["text"] == "It evicts the pods."
