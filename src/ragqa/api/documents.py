"""Document upload, versions, reads and deletion (stories A1 to A6, B1 to B4)."""

import hashlib
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Response, UploadFile
from pydantic import ValidationError

from ragqa.api.deps import ResourcesDep, SettingsDep
from ragqa.api.resources import Resources
from ragqa.api.schemas import (
    Accepted,
    BuildOut,
    DocumentDetail,
    DocumentOut,
    NoChanges,
    VersionOut,
)
from ragqa.domain.documents import DocumentMetadata
from ragqa.domain.indexes import Index
from ragqa.domain.jobs import IngestJob
from ragqa.ingestion.deletion import purge_document
from ragqa.ingestion.parsers import Parser
from ragqa.observability.logging import get_logger
from ragqa.settings import Settings
from ragqa.storage.postgres.registry import Registry, Upload

router = APIRouter(tags=["documents"])
log = get_logger(__name__)

FileField = Annotated[UploadFile, File(description="A .md or .docx file")]
MetadataField = Annotated[
    str, Form(description="JSON: doc_type, tags, author, doc_date, language (all optional)")
]
IndexField = Annotated[str | None, Form(description="Index name; see GET /indexes")]


async def _read_file(
    file: UploadFile, resources: Resources, settings: Settings
) -> tuple[Parser, bytes]:
    if not file.filename:
        raise HTTPException(422, "the file has no name")
    # Raises UnsupportedFormatError (415) before the body is read.
    parser = resources.backends.parsers.for_filename(file.filename)
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"the file is larger than {settings.max_upload_bytes} bytes")
    if not data:
        raise HTTPException(422, "the file is empty")
    return parser, data


def _parse_metadata(raw: str) -> dict[str, Any]:
    try:
        parsed = DocumentMetadata.model_validate_json(raw or "{}")
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise HTTPException(422, f"invalid metadata: {problems}") from exc
    # Only what the uploader set, so it overrides the document's own values and nothing else.
    return parsed.model_dump(mode="json", exclude_unset=True)


async def _resolve_index(registry: Registry, name: str | None, default: str | None) -> Index:
    if name or default:
        return await registry.get_index((name or default) or "")
    indexes = await registry.list_indexes()
    if len(indexes) == 1:
        return indexes[0]
    names = ", ".join(i.name for i in indexes) or "none exist yet"
    raise HTTPException(422, f"name an index with the `index` field (available: {names})")


async def _save(
    resources: Resources,
    document_id: uuid.UUID,
    file: UploadFile,
    parser: Parser,
    data: bytes,
    metadata: dict[str, Any],
) -> Upload:
    version_id = uuid.uuid4()
    filename = file.filename or ""
    raw_path = resources.backends.files.path_for(document_id, version_id, filename)
    await resources.backends.files.save(raw_path, data)
    return Upload(
        version_id=version_id,
        filename=filename,
        source_format=parser.format_key,
        content_hash=hashlib.sha256(data).hexdigest(),
        raw_path=raw_path,
        size_bytes=len(data),
        metadata=metadata,
    )


async def _publish(resources: Resources, job: IngestJob) -> None:
    """The job row is committed, so a failed publish is not an error for the client: the
    worker's republisher sends jobs that stay `queued` (ADR-0011)."""
    try:
        await resources.publisher.publish(job)
    except Exception:
        log.warning("job_publish_failed", job_id=str(job.job_id), exc_info=True)


@router.post("/documents", status_code=202)
async def upload_document(
    resources: ResourcesDep,
    settings: SettingsDep,
    file: FileField,
    title: Annotated[str | None, Form()] = None,
    metadata: MetadataField = "{}",
    index: IndexField = None,
) -> Accepted:
    """Upload a new document. Ingestion runs in the background, see `GET /jobs/{id}`."""
    registry = resources.backends.registry
    parser, data = await _read_file(file, resources, settings)
    meta = _parse_metadata(metadata)
    target = await _resolve_index(registry, index, settings.default_index)

    document_id = uuid.uuid4()
    upload = await _save(resources, document_id, file, parser, data, meta)
    try:
        job = await registry.create_document(
            document_id, title or Path(upload.filename).stem, upload, target.id
        )
    except BaseException:
        await resources.backends.files.delete(upload.raw_path)
        raise
    await _publish(resources, job)
    return Accepted(
        document_id=document_id, version=job.version, version_id=job.version_id, job_id=job.job_id
    )


@router.post("/documents/{document_id}/versions", status_code=202)
async def upload_version(
    document_id: uuid.UUID,
    resources: ResourcesDep,
    settings: SettingsDep,
    response: Response,
    file: FileField,
    metadata: MetadataField = "{}",
    index: IndexField = None,
) -> Accepted | NoChanges:
    """Upload the next version. Identical content to the newest version is not a new version."""
    registry = resources.backends.registry
    parser, data = await _read_file(file, resources, settings)
    meta = _parse_metadata(metadata)
    target = await _resolve_index(registry, index, settings.default_index)

    upload = await _save(resources, document_id, file, parser, data, meta)
    try:
        job = await registry.add_version(document_id, upload, target.id)
    except BaseException:
        await resources.backends.files.delete(upload.raw_path)
        raise
    if job is None:
        await resources.backends.files.delete(upload.raw_path)
        response.status_code = 200
        return NoChanges(document_id=document_id)
    await _publish(resources, job)
    return Accepted(
        document_id=document_id, version=job.version, version_id=job.version_id, job_id=job.job_id
    )


@router.get("/documents")
async def list_documents(
    resources: ResourcesDep, include_deleted: bool = False
) -> list[DocumentOut]:
    rows = await resources.backends.registry.list_documents(include_deleted=include_deleted)
    return [DocumentOut.model_validate(r) for r in rows]


@router.get("/documents/{document_id}")
async def get_document(document_id: uuid.UUID, resources: ResourcesDep) -> DocumentDetail:
    registry = resources.backends.registry
    document = await registry.get_document(document_id)
    latest = None
    builds: list[BuildOut] = []
    if document.latest_version_id is not None:
        latest = await registry.get_version(document.latest_version_id)
        builds = [BuildOut.model_validate(b) for b in await registry.list_builds(latest.id)]
    return DocumentDetail(
        **DocumentOut.model_validate(document).model_dump(),
        latest_version=VersionOut.model_validate(latest) if latest else None,
        builds=builds,
    )


@router.get("/documents/{document_id}/versions")
async def list_versions(document_id: uuid.UUID, resources: ResourcesDep) -> list[VersionOut]:
    registry = resources.backends.registry
    await registry.get_document(document_id)  # 404 for an unknown document
    return [VersionOut.model_validate(v) for v in await registry.list_versions(document_id)]


async def _purge(resources: Resources, document_id: uuid.UUID) -> None:
    backends = resources.backends
    try:
        await purge_document(backends.registry, backends.files, backends.chunk_store, document_id)
    except Exception:
        # The worker's maintenance loop finishes it.
        log.exception("document_purge_failed", document_id=str(document_id))


@router.delete("/documents/{document_id}", status_code=202)
async def delete_document(
    document_id: uuid.UUID, resources: ResourcesDep, background: BackgroundTasks
) -> dict[str, str]:
    """Hide the document at once, then remove its data. A tombstone row stays."""
    await resources.backends.registry.tombstone(document_id)
    background.add_task(_purge, resources, document_id)
    return {"status": "deleting"}
