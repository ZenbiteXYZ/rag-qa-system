"""Registry against a real Postgres. The tests reset the schema, so the database name must
contain "test".

    RAGQA_TEST_POSTGRES_DSN=postgresql+asyncpg://ragqa:<pw>@localhost:5432/ragqa_test \\
        uv run pytest -m integration
"""

import os
import uuid
from datetime import timedelta

import pytest
from alembic import command

from ragqa.domain.conversation import ChatMessage, NewMessage, ToolCall
from ragqa.domain.indexes import IndexConfig
from ragqa.domain.jobs import JobStatus
from ragqa.storage.postgres.migrate import alembic_config
from ragqa.storage.postgres.registry import (
    DuplicateIndexError,
    NotFoundError,
    Registry,
    Upload,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RAGQA_TEST_POSTGRES_DSN") is None, reason="RAGQA_TEST_POSTGRES_DSN not set"
    ),
]


def upload(content_hash: str = "a" * 64) -> Upload:
    version_id = uuid.uuid4()
    return Upload(
        version_id=version_id,
        filename="guide.md",
        source_format="md",
        content_hash=content_hash,
        raw_path=f"doc/{version_id}/guide.md",
        size_bytes=10,
        metadata={"tags": ["x"]},
    )


async def make_index(registry: Registry, name: str = "baseline") -> int:
    config = IndexConfig(name=name, dense_model="m", dense_vector_size=4, chunking_strategy="fixed")
    return (await registry.create_index(config)).id


def test_migrations_match_the_models(migrated: str) -> None:
    command.check(alembic_config(migrated))  # raises if the models and the database differ


async def test_indexes(registry: Registry) -> None:
    index_id = await make_index(registry)

    index = await registry.get_index("baseline")
    assert index.id == index_id
    assert index.collection == "chunks_baseline"
    assert (await registry.get_index_by_id(index_id)).name == "baseline"
    with pytest.raises(DuplicateIndexError):
        await make_index(registry)
    with pytest.raises(NotFoundError):
        await registry.get_index("missing")


async def test_versions_and_identical_content(registry: Registry) -> None:
    index_id = await make_index(registry)
    document_id = uuid.uuid4()

    first = await registry.create_document(document_id, "Guide", upload("a" * 64), index_id)
    same = await registry.add_version(document_id, upload("a" * 64), index_id)
    second = await registry.add_version(document_id, upload("b" * 64), index_id)

    assert (first.version, first.index_id, first.document_id) == (1, index_id, document_id)
    assert same is None  # story B1
    assert second is not None and second.version == 2
    versions = await registry.list_versions(document_id)
    assert [v.number for v in versions] == [1, 2]
    assert versions[0].metadata == {"tags": ["x"]}
    job = await registry.get_job(first.job_id)
    assert (job.status, job.attempts) == (JobStatus.QUEUED, 0)


async def test_job_lifecycle_and_latest_pointer(registry: Registry) -> None:
    index_id = await make_index(registry)
    document_id = uuid.uuid4()
    v1 = await registry.create_document(document_id, "Guide", upload("1" * 64), index_id)

    started = await registry.start_attempt(v1.job_id)
    await registry.set_status(v1.job_id, JobStatus.EMBEDDING)
    moved = await registry.complete_build(v1.job_id, chunk_count=7)

    assert started.attempts == 1 and started.status is JobStatus.PARSING
    assert moved is True
    assert (await registry.get_document(document_id)).latest_version_id == v1.version_id
    assert (await registry.get_job(v1.job_id)).status is JobStatus.READY
    with pytest.raises(ValueError):
        await registry.set_status(v1.job_id, JobStatus.READY)


async def test_an_older_version_never_takes_the_pointer_back(registry: Registry) -> None:
    index_id = await make_index(registry)
    document_id = uuid.uuid4()
    await registry.create_document(document_id, "Guide", upload("1" * 64), index_id)
    v2 = await registry.add_version(document_id, upload("2" * 64), index_id)
    v3 = await registry.add_version(document_id, upload("3" * 64), index_id)
    assert v2 is not None and v3 is not None

    assert await registry.complete_build(v3.job_id, chunk_count=1) is True
    assert await registry.complete_build(v2.job_id, chunk_count=1) is False
    assert (await registry.get_document(document_id)).latest_version_id == v3.version_id


async def test_stuck_jobs(registry: Registry) -> None:
    index_id = await make_index(registry)
    job = await registry.create_document(uuid.uuid4(), "Guide", upload(), index_id)

    assert [j.job_id for j in await registry.stuck_jobs(timedelta(0))] == [job.job_id]
    assert await registry.stuck_jobs(timedelta(hours=1)) == []
    await registry.start_attempt(job.job_id)
    assert await registry.stuck_jobs(timedelta(0)) == []


async def test_tombstone_and_purge(registry: Registry) -> None:
    index_id = await make_index(registry)
    document_id = uuid.uuid4()
    job = await registry.create_document(document_id, "Guide", upload(), index_id)
    await registry.complete_build(job.job_id, chunk_count=1)

    versions = await registry.tombstone(document_id)
    await registry.purge_versions(document_id)

    assert [v.id for v in versions] == [job.version_id]
    document = await registry.get_document(document_id)
    assert document.deleted_at is not None and document.latest_version_id is None
    assert await registry.list_versions(document_id) == []
    assert await registry.list_documents() == []
    assert len(await registry.list_documents(include_deleted=True)) == 1
    with pytest.raises(NotFoundError):
        await registry.get_job(job.job_id)  # jobs go with their versions
    with pytest.raises(NotFoundError):
        await registry.add_version(document_id, upload("c" * 64), index_id)


async def test_conversation_keeps_order_and_tool_turns(registry: Registry) -> None:
    conversation_id = await registry.create_conversation()
    call = ToolCall(id="c1", name="search_docs", arguments={"q": "ivan"})
    await registry.append_messages(
        conversation_id,
        [
            NewMessage(message=ChatMessage(role="user", content="Who wrote the guide?")),
            NewMessage(message=ChatMessage(role="assistant", tool_calls=(call,))),
            NewMessage(message=ChatMessage(role="tool", content="[]", tool_call_id="c1")),
        ],
    )
    stored = await registry.append_messages(
        conversation_id,
        [
            NewMessage(
                message=ChatMessage(role="assistant", content="I. Petrov."),
                trace_id="tr-1",
                citations=({"title": "Guide", "chunk_index": 3},),
            )
        ],
    )
    assert stored[0].position == 3

    history = await registry.get_messages(conversation_id)
    assert [m.position for m in history] == [0, 1, 2, 3]
    assert history[1].message.tool_calls == (call,)
    assert history[2].message.tool_call_id == "c1"
    assert history[3].trace_id == "tr-1"
    assert history[3].citations == ({"title": "Guide", "chunk_index": 3},)

    newest = await registry.get_messages(conversation_id, last=2)
    assert [m.position for m in newest] == [2, 3]


async def test_unknown_conversation(registry: Registry) -> None:
    with pytest.raises(NotFoundError):
        await registry.get_messages(uuid.uuid4())
    with pytest.raises(NotFoundError):
        await registry.append_messages(
            uuid.uuid4(), [NewMessage(message=ChatMessage(role="user", content="hi"))]
        )


async def test_complete_build_reports_latest_again_after_a_retry(registry: Registry) -> None:
    index_id = await make_index(registry)
    document_id = uuid.uuid4()
    job = await registry.create_document(document_id, "Guide", upload(), index_id)

    assert await registry.complete_build(job.job_id, chunk_count=3) is True
    # The worker crashed before flipping `is_latest` in Qdrant and the job came back.
    assert await registry.complete_build(job.job_id, chunk_count=3) is True


async def test_same_version_built_in_a_second_index_flips_there_too(registry: Registry) -> None:
    first, second = await make_index(registry, "a"), await make_index(registry, "b")
    document_id = uuid.uuid4()
    job_a = await registry.create_document(document_id, "Guide", upload(), first)
    job_b = await registry.add_version(document_id, upload("b" * 64), second)
    assert job_b is not None

    assert await registry.complete_build(job_a.job_id, chunk_count=1) is True
    assert await registry.complete_build(job_b.job_id, chunk_count=1) is True  # newer, takes it
    # Version 1 finishing late anywhere never takes the pointer back.
    late = await registry.add_version(document_id, upload("c" * 64), first)
    assert late is not None
    assert await registry.complete_build(late.job_id, chunk_count=1) is True
    assert await registry.complete_build(job_a.job_id, chunk_count=1) is False


async def test_list_builds_and_pending_deletions(registry: Registry) -> None:
    index_id = await make_index(registry, "base")
    document_id = uuid.uuid4()
    job = await registry.create_document(document_id, "Guide", upload(), index_id)
    await registry.complete_build(job.job_id, chunk_count=7)

    builds = await registry.list_builds(job.version_id)
    assert [(b.index_name, b.status, b.chunk_count) for b in builds] == [
        ("base", JobStatus.READY, 7)
    ]

    assert await registry.pending_deletions() == []
    await registry.tombstone(document_id)
    assert await registry.pending_deletions() == [document_id]
    await registry.purge_versions(document_id)
    assert await registry.pending_deletions() == []
