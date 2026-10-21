"""Ingestion entry point called by the worker for each job."""

from ragqa.domain.jobs import IngestJob
from ragqa.observability.logging import get_logger

log = get_logger(__name__)


async def ingest_document(job: IngestJob) -> None:
    """Parse, chunk, embed and upsert one document version, then flip `is_latest`.

    Stub until slices 2 and 3. Must be idempotent, since delivery is at-least-once
    (ADR-0011, ADR-0014). Raise `PermanentIngestionError` for inputs that retrying
    cannot fix.
    """
    log.warning("ingest_stub_called", index_id=job.index_id)
