"""Ingestion of one document version into one index (ADR-0011, ADR-0014).

Steps: parse, chunk, embed, upsert with `is_latest=false`, mark the build ready in Postgres
(which may move the latest pointer), then flip `is_latest` in Qdrant. Every step can be
repeated: point ids are deterministic and `complete_build` never moves the pointer backwards,
so at-least-once delivery is safe.

Errors that retrying cannot fix are raised as `PermanentIngestionError`. Anything else
(embedder down, Qdrant or Postgres unavailable) propagates and the worker retries the job.
"""

import asyncio
from typing import Any

from pydantic import ValidationError

from ragqa.domain.documents import (
    Chunk,
    DocumentMetadata,
    EmbeddedChunk,
    ParsedDocument,
    check_chunks,
)
from ragqa.domain.indexes import Index
from ragqa.domain.jobs import IngestJob, JobStatus
from ragqa.embeddings.tei import Embedder, EmbedderError, EmbedderUnavailable
from ragqa.ingestion.chunkers import ChunkerRegistry, TokenCounterFactory
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers import ParserRegistry
from ragqa.observability.logging import get_logger
from ragqa.observability.tracing import traced_stage
from ragqa.storage.files.store import FileStore
from ragqa.storage.postgres.registry import NotFoundError, Registry, VersionInfo
from ragqa.storage.qdrant.store import ChunkStore, VersionRef

log = get_logger(__name__)


def merged_metadata(parsed: DocumentMetadata, uploaded: dict[str, Any]) -> DocumentMetadata:
    """Metadata from the document itself, overridden by what the uploader sent."""
    values = {k: v for k, v in parsed.model_dump().items() if v not in (None, ())}
    values.update(uploaded)
    return DocumentMetadata.model_validate(values)


class IngestionService:
    def __init__(
        self,
        *,
        registry: Registry,
        files: FileStore,
        chunk_store: ChunkStore,
        embedder: Embedder,
        parsers: ParserRegistry,
        chunkers: ChunkerRegistry,
        token_counter: TokenCounterFactory | None,
    ) -> None:
        self._registry = registry
        self._files = files
        self._chunk_store = chunk_store
        self._embedder = embedder
        self._parsers = parsers
        self._chunkers = chunkers
        self._token_counter = token_counter

    async def ingest(self, job: IngestJob) -> None:
        try:
            await self._ingest(job)
        except NotFoundError as exc:
            # The job, version or document is gone, for example after a delete.
            raise PermanentIngestionError(str(exc)) from exc

    async def _ingest(self, job: IngestJob) -> None:
        await self._registry.start_attempt(job.job_id)
        version = await self._registry.get_version(job.version_id)
        document = await self._registry.get_document(version.document_id)
        if document.deleted_at is not None:
            raise PermanentIngestionError(f"document {document.id} was deleted")
        index = await self._registry.get_index_by_id(job.index_id)

        with traced_stage("ingest_parse"):
            parsed = await self._parse(version)
        await self._registry.set_status(job.job_id, JobStatus.CHUNKING)
        with traced_stage("ingest_chunk"):
            chunks = await self._chunk(index, parsed)
        await self._registry.set_status(job.job_id, JobStatus.EMBEDDING)
        with traced_stage("ingest_embed"):
            embedded = await self._embed(index, chunks)

        try:
            metadata = merged_metadata(parsed.metadata, version.metadata)
        except ValidationError as exc:
            raise PermanentIngestionError(f"invalid document metadata: {exc}") from exc
        ref = VersionRef(
            document_id=document.id,
            version_id=version.id,
            version=version.number,
            title=document.title,
            metadata=metadata,
        )
        with traced_stage("ingest_upsert"):
            await self._chunk_store.ensure_collection(index)
            count = await self._chunk_store.upsert(index, ref, embedded)

        # A delete that started while this job ran must not leave points behind.
        if (await self._registry.get_document(document.id)).deleted_at is not None:
            await self._chunk_store.delete_document(index, document.id)
            raise PermanentIngestionError(f"document {document.id} was deleted during ingestion")

        is_latest = await self._registry.complete_build(job.job_id, chunk_count=count)
        if is_latest:
            await self._chunk_store.set_latest(index, document.id, version.id)
        log.info("ingest_done", chunks=count, is_latest=is_latest, index=index.name)

    async def _parse(self, version: VersionInfo) -> ParsedDocument:
        try:
            data = await self._files.read(version.raw_path)
        except FileNotFoundError as exc:
            raise PermanentIngestionError(f"raw file {version.raw_path} is missing") from exc
        parser = self._parsers.for_filename(version.filename)
        # CPU-bound, so off the event loop (ADR-0011).
        return await asyncio.to_thread(parser.parse, data, filename=version.filename)

    async def _chunk(self, index: Index, parsed: ParsedDocument) -> list[Chunk]:
        if self._token_counter is None:
            raise PermanentIngestionError("no token counter is configured for the chunkers")
        try:
            # The counter may load a tokenizer and the chunker is CPU-bound: both off the loop.
            chunks = await asyncio.to_thread(self._chunk_sync, self._token_counter, index, parsed)
        except (
            ValueError
        ) as exc:  # unknown strategy, bad params, or a chunker breaking its contract
            raise PermanentIngestionError(f"chunking failed: {exc}") from exc
        return chunks

    def _chunk_sync(
        self, make_counter: TokenCounterFactory, index: Index, parsed: ParsedDocument
    ) -> list[Chunk]:
        chunker = self._chunkers.build(
            index.chunking_strategy, index.chunking_params, make_counter(index)
        )
        chunks = chunker.chunk(parsed)
        check_chunks(parsed, chunks)
        return chunks

    async def _embed(self, index: Index, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        try:
            vectors = await self._embedder.embed_passages([c.text for c in chunks])
            return [
                EmbeddedChunk.of(chunk, vector, size=index.dense_vector_size)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
        except EmbedderUnavailable:
            raise
        except (EmbedderError, ValueError) as exc:
            # The input was rejected, or the vector size does not match the index.
            raise PermanentIngestionError(f"embedding failed: {exc}") from exc
