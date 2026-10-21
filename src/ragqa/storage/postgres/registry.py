"""Registry operations on Postgres. Each public method is one transaction (ADR-0014).

Postgres is the source of truth for which version is latest. Qdrant's `is_latest` payload
is a copy that the worker updates after `complete_build` returns.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ragqa.domain.conversation import ChatMessage, NewMessage, StoredMessage, ToolCall
from ragqa.domain.indexes import Index, IndexConfig, collection_for
from ragqa.domain.jobs import IngestJob, JobStatus
from ragqa.storage.postgres.models import (
    Conversation,
    Document,
    IndexRecord,
    Job,
    Message,
    Version,
    VersionIndexBuild,
)


class RegistryError(Exception):
    """Base class for registry errors the API turns into client errors."""


class DuplicateIndexError(RegistryError):
    pass


class NotFoundError(RegistryError):
    pass


@dataclass(frozen=True)
class Upload:
    """A stored raw file, ready to become a version."""

    version_id: uuid.UUID
    filename: str
    source_format: str
    content_hash: str
    raw_path: str
    size_bytes: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentInfo:
    id: uuid.UUID
    title: str
    latest_version_id: uuid.UUID | None
    created_at: datetime
    deleted_at: datetime | None


@dataclass(frozen=True)
class VersionInfo:
    id: uuid.UUID
    document_id: uuid.UUID
    number: int
    filename: str
    source_format: str
    content_hash: str
    raw_path: str
    size_bytes: int
    metadata: dict[str, Any]
    uploaded_at: datetime


@dataclass(frozen=True)
class JobInfo:
    id: uuid.UUID
    version_id: uuid.UUID
    index_id: int
    status: JobStatus
    attempts: int
    error: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BuildInfo:
    index_name: str
    status: JobStatus
    chunk_count: int | None
    built_at: datetime | None


def create_session_factory(dsn: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(dsn, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _index(row: IndexRecord) -> Index:
    return Index(
        id=row.id,
        collection=row.collection,
        name=row.name,
        dense_model=row.dense_model,
        dense_vector_size=row.dense_vector_size,
        dense_distance=row.dense_distance,  # type: ignore[arg-type]  # the check constraint keeps it valid
        sparse_model=row.sparse_model,
        chunking_strategy=row.chunking_strategy,
        chunking_params=row.chunking_params,
    )


def _document(row: Document) -> DocumentInfo:
    return DocumentInfo(row.id, row.title, row.latest_version_id, row.created_at, row.deleted_at)


def _version(row: Version) -> VersionInfo:
    return VersionInfo(
        id=row.id,
        document_id=row.document_id,
        number=row.number,
        filename=row.filename,
        source_format=row.source_format,
        content_hash=row.content_hash,
        raw_path=row.raw_path,
        size_bytes=row.size_bytes,
        metadata=row.doc_metadata,
        uploaded_at=row.uploaded_at,
    )


def _job(row: Job) -> JobInfo:
    return JobInfo(
        row.id,
        row.version_id,
        row.index_id,
        JobStatus(row.status),
        row.attempts,
        row.error,
        row.created_at,
        row.updated_at,
    )


def _stored(row: Message) -> StoredMessage:
    return StoredMessage(
        id=row.id,
        conversation_id=row.conversation_id,
        position=row.position,
        created_at=row.created_at,
        trace_id=row.trace_id,
        citations=tuple(row.citations),
        message=ChatMessage(
            role=row.role,  # type: ignore[arg-type]
            content=row.content,
            tool_calls=tuple(ToolCall(**call) for call in row.tool_calls),
            tool_call_id=row.tool_call_id,
        ),
    )


def _message(job: Job, version: Version) -> IngestJob:
    return IngestJob(
        job_id=job.id,
        document_id=version.document_id,
        version_id=version.id,
        version=version.number,
        index_id=job.index_id,
    )


class Registry:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    # --- Indexes

    async def create_index(self, config: IndexConfig) -> Index:
        row = IndexRecord(collection=collection_for(config.name), **config.model_dump())
        try:
            async with self._sessions.begin() as session:
                session.add(row)
        except IntegrityError as exc:
            raise DuplicateIndexError(f"an index named {config.name!r} already exists") from exc
        return _index(row)

    async def get_index(self, name: str) -> Index:
        async with self._sessions() as session:
            row = await session.scalar(select(IndexRecord).where(IndexRecord.name == name))
        if row is None:
            raise NotFoundError(f"no index named {name!r}")
        return _index(row)

    async def get_index_by_id(self, index_id: int) -> Index:
        async with self._sessions() as session:
            row = await session.get(IndexRecord, index_id)
        if row is None:
            raise NotFoundError(f"no index with id {index_id}")
        return _index(row)

    async def list_indexes(self) -> list[Index]:
        async with self._sessions() as session:
            rows = await session.scalars(select(IndexRecord).order_by(IndexRecord.id))
            return [_index(r) for r in rows]

    # --- Documents and versions

    async def create_document(
        self, document_id: uuid.UUID, title: str, upload: Upload, index_id: int
    ) -> IngestJob:
        """A new document with version 1 and a queued job, in one transaction."""
        async with self._sessions.begin() as session:
            session.add(Document(id=document_id, title=title))
            await session.flush()  # the version's foreign key needs the document row
            return await self._add_version(session, document_id, 1, upload, index_id)

    async def add_version(
        self, document_id: uuid.UUID, upload: Upload, index_id: int
    ) -> IngestJob | None:
        """The next version and a queued job. None if the content equals the newest version."""
        async with self._sessions.begin() as session:
            # Locking the document row serializes concurrent uploads to the same document.
            document = await session.get(Document, document_id, with_for_update=True)
            if document is None or document.deleted_at is not None:
                raise NotFoundError(f"no document {document_id}")
            newest = await session.scalar(
                select(Version)
                .where(Version.document_id == document_id)
                .order_by(Version.number.desc())
                .limit(1)
            )
            if newest is not None and newest.content_hash == upload.content_hash:
                return None  # story B1: identical content is not a new version
            number = newest.number + 1 if newest is not None else 1
            return await self._add_version(session, document_id, number, upload, index_id)

    async def _add_version(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
        number: int,
        upload: Upload,
        index_id: int,
    ) -> IngestJob:
        version = Version(
            id=upload.version_id,
            document_id=document_id,
            number=number,
            filename=upload.filename,
            source_format=upload.source_format,
            content_hash=upload.content_hash,
            raw_path=upload.raw_path,
            size_bytes=upload.size_bytes,
            doc_metadata=upload.metadata,
        )
        job = Job(id=uuid.uuid4(), version_id=version.id, index_id=index_id)
        session.add(version)
        await session.flush()
        session.add(job)
        session.add(
            VersionIndexBuild(version_id=version.id, index_id=index_id, status=JobStatus.QUEUED)
        )
        await session.flush()
        return _message(job, version)

    async def get_document(self, document_id: uuid.UUID) -> DocumentInfo:
        async with self._sessions() as session:
            row = await session.get(Document, document_id)
        if row is None:
            raise NotFoundError(f"no document {document_id}")
        return _document(row)

    async def list_documents(self, *, include_deleted: bool = False) -> list[DocumentInfo]:
        query = select(Document).order_by(Document.created_at)
        if not include_deleted:
            query = query.where(Document.deleted_at.is_(None))
        async with self._sessions() as session:
            return [_document(r) for r in await session.scalars(query)]

    async def list_versions(self, document_id: uuid.UUID) -> list[VersionInfo]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(Version).where(Version.document_id == document_id).order_by(Version.number)
            )
            return [_version(r) for r in rows]

    async def get_version(self, version_id: uuid.UUID) -> VersionInfo:
        async with self._sessions() as session:
            row = await session.get(Version, version_id)
        if row is None:
            raise NotFoundError(f"no version {version_id}")
        return _version(row)

    # --- Jobs

    async def get_job(self, job_id: uuid.UUID) -> JobInfo:
        async with self._sessions() as session:
            row = await session.get(Job, job_id)
        if row is None:
            raise NotFoundError(f"no job {job_id}")
        return _job(row)

    async def start_attempt(self, job_id: uuid.UUID) -> JobInfo:
        """Count an attempt and move the job back to `parsing` (a retry starts over)."""
        async with self._sessions.begin() as session:
            job = await session.get(Job, job_id, with_for_update=True)
            if job is None:
                raise NotFoundError(f"no job {job_id}")
            job.attempts += 1
            job.status = JobStatus.PARSING
            job.error = None
            await self._set_build(session, job, JobStatus.PARSING)
            await session.flush()
            await session.refresh(job)
            return _job(job)

    async def set_status(
        self, job_id: uuid.UUID, status: JobStatus, *, error: str | None = None
    ) -> None:
        """Progress (`chunking`, `embedding`) or a final failure (`failed`, `dead`)."""
        if status is JobStatus.READY:
            raise ValueError("use complete_build to finish a job")
        async with self._sessions.begin() as session:
            job = await session.get(Job, job_id, with_for_update=True)
            if job is None:
                raise NotFoundError(f"no job {job_id}")
            job.status = status
            job.error = error
            await self._set_build(session, job, status)

    async def complete_build(self, job_id: uuid.UUID, *, chunk_count: int) -> bool:
        """ADR-0014 step 2: build ready, job ready, and the latest pointer moved if this is
        the newest version that is ready anywhere. Returns True if this version is the latest
        one, so the caller then flips `is_latest` in Qdrant (step 3). The flip is idempotent."""
        async with self._sessions.begin() as session:
            job = await session.get(Job, job_id, with_for_update=True)
            if job is None:
                raise NotFoundError(f"no job {job_id}")
            version = await session.get(Version, job.version_id)
            assert version is not None  # foreign key
            document = await session.get(Document, version.document_id, with_for_update=True)
            assert document is not None
            job.status = JobStatus.READY
            job.error = None
            await self._set_build(session, job, JobStatus.READY, chunk_count=chunk_count)

            current_number = None
            if document.latest_version_id is not None:
                current_number = await session.scalar(
                    select(Version.number).where(Version.id == document.latest_version_id)
                )
            # A newer version may have finished first; an older one never takes the pointer back.
            if current_number is None or version.number > current_number:
                document.latest_version_id = version.id
                return True
            # A retry after a crash between this commit and the Qdrant flip, or the same
            # version built in a second index: the pointer is already here, the flip is not.
            return document.latest_version_id == version.id

    async def stuck_jobs(self, older_than: timedelta) -> list[IngestJob]:
        """Jobs still `queued` after `older_than`, for the republisher (ADR-0011)."""
        cutoff = datetime.now(UTC) - older_than
        async with self._sessions() as session:
            rows = await session.execute(
                select(Job, Version)
                .join(Version, Version.id == Job.version_id)
                .where(Job.status == JobStatus.QUEUED, Job.updated_at < cutoff)
                .order_by(Job.updated_at)
            )
            return [_message(job, version) for job, version in rows]

    async def _set_build(
        self,
        session: AsyncSession,
        job: Job,
        status: JobStatus,
        *,
        chunk_count: int | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status.value}
        if status is JobStatus.READY:
            values |= {"chunk_count": chunk_count, "built_at": func.now()}
        await session.execute(
            insert(VersionIndexBuild)
            .values(version_id=job.version_id, index_id=job.index_id, **values)
            .on_conflict_do_update(index_elements=["version_id", "index_id"], set_=values)
        )

    async def list_builds(self, version_id: uuid.UUID) -> list[BuildInfo]:
        async with self._sessions() as session:
            rows = await session.execute(
                select(IndexRecord.name, VersionIndexBuild)
                .join(IndexRecord, IndexRecord.id == VersionIndexBuild.index_id)
                .where(VersionIndexBuild.version_id == version_id)
                .order_by(IndexRecord.name)
            )
            return [
                BuildInfo(name, JobStatus(b.status), b.chunk_count, b.built_at) for name, b in rows
            ]

    # --- Deletion (Flow 7)

    async def tombstone(self, document_id: uuid.UUID) -> list[VersionInfo]:
        """Step 1: hide the document. Returns its versions, for cleaning up vectors and files."""
        async with self._sessions.begin() as session:
            document = await session.get(Document, document_id, with_for_update=True)
            if document is None:
                raise NotFoundError(f"no document {document_id}")
            if document.deleted_at is None:
                document.deleted_at = func.now()
            rows = await session.scalars(select(Version).where(Version.document_id == document_id))
            return [_version(r) for r in rows]

    async def pending_deletions(self) -> list[uuid.UUID]:
        """Tombstoned documents whose versions are still there, for the cleanup loop."""
        async with self._sessions() as session:
            rows = await session.scalars(
                select(Document.id)
                .where(
                    Document.deleted_at.is_not(None),
                    select(Version.id).where(Version.document_id == Document.id).exists(),
                )
                .order_by(Document.deleted_at)
            )
            return list(rows)

    async def purge_versions(self, document_id: uuid.UUID) -> None:
        """Step 3, after vectors and files are gone: drop versions, keep the tombstone row.
        Builds and jobs go with the versions (ON DELETE CASCADE)."""
        async with self._sessions.begin() as session:
            await session.execute(
                update(Document).where(Document.id == document_id).values(latest_version_id=None)
            )
            for version in await session.scalars(
                select(Version).where(Version.document_id == document_id)
            ):
                await session.delete(version)

    # --- Conversations (story C4)

    async def create_conversation(self) -> uuid.UUID:
        conversation_id = uuid.uuid4()
        async with self._sessions.begin() as session:
            session.add(Conversation(id=conversation_id))
        return conversation_id

    async def append_messages(
        self, conversation_id: uuid.UUID, items: Sequence[NewMessage]
    ) -> list[StoredMessage]:
        """Add messages after the existing ones, in one transaction.

        The conversation row is locked, so two requests to the same conversation get
        distinct positions.
        """
        async with self._sessions.begin() as session:
            conversation = await session.get(Conversation, conversation_id, with_for_update=True)
            if conversation is None:
                raise NotFoundError(f"no conversation {conversation_id}")
            last = await session.scalar(
                select(func.max(Message.position)).where(Message.conversation_id == conversation_id)
            )
            rows = [
                Message(
                    id=uuid.uuid4(),
                    conversation_id=conversation_id,
                    position=(-1 if last is None else last) + 1 + offset,
                    role=item.message.role,
                    content=item.message.content,
                    tool_calls=[call.model_dump() for call in item.message.tool_calls],
                    tool_call_id=item.message.tool_call_id,
                    citations=list(item.citations),
                    trace_id=item.trace_id,
                )
                for offset, item in enumerate(items)
            ]
            session.add_all(rows)
            conversation.updated_at = datetime.now(UTC)
            await session.flush()
            await session.refresh(conversation)
            for row in rows:
                await session.refresh(row)
            return [_stored(row) for row in rows]

    async def get_messages(
        self, conversation_id: uuid.UUID, *, last: int | None = None
    ) -> list[StoredMessage]:
        """Oldest first. `last` keeps only the newest n messages."""
        async with self._sessions() as session:
            if await session.get(Conversation, conversation_id) is None:
                raise NotFoundError(f"no conversation {conversation_id}")
            query = select(Message).where(Message.conversation_id == conversation_id)
            if last is None:
                rows = (await session.scalars(query.order_by(Message.position))).all()
            else:
                newest = query.order_by(Message.position.desc()).limit(last)
                rows = list(reversed((await session.scalars(newest)).all()))
        return [_stored(row) for row in rows]
