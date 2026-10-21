"""Ingestion of a document version through the real registry and Qdrant, with a fake embedder.

    RAGQA_TEST_POSTGRES_DSN=...ragqa_test RAGQA_TEST_QDRANT_URL=http://localhost:6333 \\
        uv run pytest -m integration
"""

import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from ragqa.domain.indexes import Index, IndexConfig
from ragqa.domain.jobs import IngestJob, JobStatus
from ragqa.ingestion.chunkers import ChunkerRegistry
from ragqa.ingestion.chunkers import default_registry as default_chunkers
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers import default_registry
from ragqa.ingestion.service import IngestionService
from ragqa.storage.files.store import FileStore
from ragqa.storage.postgres.registry import Registry, Upload
from ragqa.storage.qdrant.store import ChunkStore
from ragqa.worker.maintenance import finish_deletions, republish_stuck
from tests.integration.conftest import QDRANT_URL
from tests.integration.fakes import (
    DIMENSIONS,
    FakeEmbedder,
    make_chunkers,
    word_counter_factory,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RAGQA_TEST_POSTGRES_DSN") is None
        or os.environ.get("RAGQA_TEST_QDRANT_URL") is None,
        reason="RAGQA_TEST_POSTGRES_DSN and RAGQA_TEST_QDRANT_URL are not both set",
    ),
]

GUIDE_V1 = b"# Drain\n\nUse kubectl drain to empty a node.\n\nIt evicts the pods.\n"
GUIDE_V2 = (
    b"# Drain\n\nUse kubectl drain to empty a node.\n\nIt evicts the pods.\n\nThen cordon it.\n"
)


@dataclass
class Stack:
    registry: Registry
    files: FileStore
    store: ChunkStore
    embedder: FakeEmbedder
    index: Index
    service: IngestionService

    def service_with(
        self, chunkers: ChunkerRegistry | None = None, *, counter: bool = True
    ) -> IngestionService:
        return IngestionService(
            registry=self.registry,
            files=self.files,
            chunk_store=self.store,
            embedder=self.embedder,
            parsers=default_registry(),
            chunkers=chunkers or make_chunkers(),
            token_counter=word_counter_factory() if counter else None,
        )

    async def upload(self, data: bytes, document_id: uuid.UUID | None = None) -> IngestJob:
        """Store the file and register it as the API does, without publishing."""
        document_id = document_id or uuid.uuid4()
        version_id = uuid.uuid4()
        raw_path = self.files.path_for(document_id, version_id, "drain.md")
        await self.files.save(raw_path, data)
        upload = Upload(
            version_id=version_id,
            filename="drain.md",
            source_format="md",
            content_hash=hashlib.sha256(data).hexdigest(),
            raw_path=raw_path,
            size_bytes=len(data),
            metadata={"author": "I. Petrov"},
        )
        if await self._exists(document_id):
            job = await self.registry.add_version(document_id, upload, self.index.id)
            assert job is not None
            return job
        return await self.registry.create_document(document_id, "Drain", upload, self.index.id)

    async def _exists(self, document_id: uuid.UUID) -> bool:
        return any(d.id == document_id for d in await self.registry.list_documents())


@pytest.fixture
async def stack(registry: Registry, tmp_path: Path) -> AsyncIterator[Stack]:
    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    store = ChunkStore(client)
    index = await registry.create_index(
        IndexConfig(
            name=f"flow_{uuid.uuid4().hex[:8]}",
            dense_model="fake",
            dense_vector_size=DIMENSIONS,
            chunking_strategy="paragraphs",
        )
    )
    embedder = FakeEmbedder()
    files = FileStore(tmp_path)
    built = Stack(registry, files, store, embedder, index, None)  # type: ignore[arg-type]
    built.service = built.service_with()
    yield built
    await client.delete_collection(index.collection)
    await client.close()


async def test_ingest_builds_the_version_and_marks_it_latest(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)

    await stack.service.ingest(job)

    info = await stack.registry.get_job(job.job_id)
    assert info.status is JobStatus.READY
    assert await stack.store.count(stack.index) == 3  # heading, two paragraphs
    assert await stack.store.count(stack.index, is_latest=True) == 3
    document = await stack.registry.get_document(job.document_id)
    assert document.latest_version_id == job.version_id
    (build,) = await stack.registry.list_builds(job.version_id)
    assert (build.status, build.chunk_count) == (JobStatus.READY, 3)


async def test_uploaded_metadata_reaches_the_payload(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)
    await stack.service.ingest(job)

    assert await stack.store.count(stack.index, author="I. Petrov") == 3


async def test_ingesting_the_same_job_again_changes_nothing(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)

    await stack.service.ingest(job)
    await stack.service.ingest(job)  # at-least-once delivery

    assert await stack.store.count(stack.index) == 3
    assert await stack.store.count(stack.index, is_latest=True) == 3
    assert (await stack.registry.get_job(job.job_id)).attempts == 2


async def test_a_retry_repairs_a_missed_is_latest_flip(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)
    await stack.service.ingest(job)
    # The worker died after Postgres committed and before the Qdrant flip.
    await stack.store.set_latest(stack.index, job.document_id, uuid.uuid4())
    assert await stack.store.count(stack.index, is_latest=True) == 0

    await stack.service.ingest(job)

    assert await stack.store.count(stack.index, is_latest=True) == 3


async def test_new_version_takes_over_is_latest(stack: Stack) -> None:
    first = await stack.upload(GUIDE_V1)
    await stack.service.ingest(first)
    second = await stack.upload(GUIDE_V2, first.document_id)
    assert second.version == 2

    await stack.service.ingest(second)

    assert await stack.store.count(stack.index, version=1, is_latest=True) == 0
    assert await stack.store.count(stack.index, version=1) == 3  # kept for version queries
    assert await stack.store.count(stack.index, version=2, is_latest=True) == 4


async def test_missing_raw_file_is_permanent(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)
    version = await stack.registry.get_version(job.version_id)
    await stack.files.delete(version.raw_path)

    with pytest.raises(PermanentIngestionError, match="missing"):
        await stack.service.ingest(job)


async def test_deleted_document_is_permanent(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)
    await stack.registry.tombstone(job.document_id)

    with pytest.raises(PermanentIngestionError, match="deleted"):
        await stack.service.ingest(job)


async def test_unknown_strategy_and_missing_counter_are_permanent(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)

    with pytest.raises(PermanentIngestionError, match="unknown chunking strategy"):
        await stack.service_with(ChunkerRegistry()).ingest(job)
    with pytest.raises(PermanentIngestionError, match="token counter"):
        await stack.service_with(counter=False).ingest(job)


async def test_job_for_a_purged_version_is_permanent(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)
    await stack.registry.purge_versions(job.document_id)

    with pytest.raises(PermanentIngestionError, match="no job"):
        await stack.service.ingest(job)


async def test_republisher_sends_stuck_jobs_once(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)
    sent: list[IngestJob] = []

    async def publish(item: IngestJob) -> None:
        sent.append(item)

    assert await republish_stuck(stack.registry, publish, timedelta(seconds=60)) == 0
    assert await republish_stuck(stack.registry, publish, timedelta(0)) == 1
    assert sent == [job]
    # `updated_at` was bumped, so it is not sent again within the window.
    assert await republish_stuck(stack.registry, publish, timedelta(seconds=60)) == 0


async def test_finish_deletions_removes_vectors_files_and_versions(stack: Stack) -> None:
    job = await stack.upload(GUIDE_V1)
    await stack.service.ingest(job)
    version = await stack.registry.get_version(job.version_id)
    await stack.registry.tombstone(job.document_id)  # the API stopped here

    assert await finish_deletions(stack.registry, stack.files, stack.store) == 1

    assert await stack.store.count(stack.index) == 0
    with pytest.raises(FileNotFoundError):
        await stack.files.read(version.raw_path)
    assert await stack.registry.list_versions(job.document_id) == []
    assert await finish_deletions(stack.registry, stack.files, stack.store) == 0
    assert [d.id for d in await stack.registry.list_documents(include_deleted=True)] == [
        job.document_id
    ]


async def test_default_chunkers_ingest_a_real_document(stack: Stack) -> None:
    for strategy in ("fixed-size", "recursive"):
        index = await stack.registry.create_index(
            IndexConfig(
                name=f"real_{strategy.replace('-', '_')}",
                dense_model="fake",
                dense_vector_size=DIMENSIONS,
                chunking_strategy=strategy,
                chunking_params={"chunk_tokens": 8, "overlap_tokens": 2}
                if strategy == "recursive"
                else {"chunk_chars": 60, "overlap_chars": 10},
            )
        )
        stack.index = index
        job = await stack.upload(GUIDE_V1 * 3)
        try:
            await stack.service_with(default_chunkers()).ingest(job)
            count = await stack.store.count(index)
            assert count > 2
            assert await stack.store.count(index, is_latest=True) == count
        finally:
            await stack.store.delete_document(index, job.document_id)
