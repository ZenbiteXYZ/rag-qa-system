"""Writes final job failures to the registry, so `GET /jobs/{id}` shows them."""

from ragqa.domain.jobs import IngestJob, JobStatus
from ragqa.storage.postgres.registry import Registry


class RegistryJobReporter:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def failed(self, job: IngestJob, error: str) -> None:
        await self._registry.set_status(job.job_id, JobStatus.FAILED, error=error)

    async def dead(self, job: IngestJob, error: str) -> None:
        await self._registry.set_status(job.job_id, JobStatus.DEAD, error=error)
