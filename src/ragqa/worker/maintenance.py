"""Housekeeping in the worker process.

- Re-send jobs that stayed `queued`: the API commits the job row and then publishes, so a
  crash between the two (or a broker outage) leaves a job nobody will pick up (ADR-0011).
- Finish deletions the API started but did not complete (Flow 7).
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta

from ragqa.domain.jobs import IngestJob, JobStatus
from ragqa.ingestion.deletion import purge_document
from ragqa.observability.logging import get_logger
from ragqa.storage.files.store import FileStore
from ragqa.storage.postgres.registry import Registry
from ragqa.storage.qdrant.store import ChunkStore

log = get_logger(__name__)

Publish = Callable[[IngestJob], Awaitable[None]]


async def republish_stuck(registry: Registry, publish: Publish, older_than: timedelta) -> int:
    jobs = await registry.stuck_jobs(older_than)
    for job in jobs:
        await publish(job)
        # Bumps `updated_at`, so the job is not sent again before `older_than` passes.
        await registry.set_status(job.job_id, JobStatus.QUEUED)
        log.warning("job_republished", job_id=str(job.job_id))
    return len(jobs)


async def finish_deletions(registry: Registry, files: FileStore, chunk_store: ChunkStore) -> int:
    pending = await registry.pending_deletions()
    for document_id in pending:
        await purge_document(registry, files, chunk_store, document_id)
    return len(pending)


async def run_maintenance(
    registry: Registry,
    files: FileStore,
    chunk_store: ChunkStore,
    publish: Publish,
    *,
    interval_s: float,
    republish_after_s: float,
    stop: asyncio.Event,
) -> None:
    """Run both tasks every `interval_s` until `stop` is set. A failed pass is logged and
    retried on the next one, so a broker or database outage does not end the loop."""
    while not stop.is_set():
        try:
            await republish_stuck(registry, publish, timedelta(seconds=republish_after_s))
            await finish_deletions(registry, files, chunk_store)
        except Exception:
            log.exception("maintenance_failed")
        try:
            await asyncio.wait_for(stop.wait(), interval_s)
        except TimeoutError:
            pass
