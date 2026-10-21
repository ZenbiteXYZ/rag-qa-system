import asyncio
import uuid
from datetime import timedelta
from typing import Any

from ragqa.domain.jobs import IngestJob, JobStatus
from ragqa.worker.maintenance import run_maintenance


class FakeRegistry:
    def __init__(self) -> None:
        self.passes = 0
        self.statuses: list[tuple[uuid.UUID, JobStatus]] = []
        self.job = IngestJob(
            job_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            version=1,
            index_id=1,
        )

    async def stuck_jobs(self, older_than: timedelta) -> list[IngestJob]:
        self.passes += 1
        if self.passes == 1:
            raise ConnectionError("database restarting")  # one failed pass must not end the loop
        return [self.job] if self.passes == 2 else []

    async def set_status(self, job_id: uuid.UUID, status: JobStatus, **_: Any) -> None:
        self.statuses.append((job_id, status))

    async def pending_deletions(self) -> list[uuid.UUID]:
        return []


async def test_loop_survives_a_failed_pass_republishes_and_stops() -> None:
    registry, sent, stop = FakeRegistry(), [], asyncio.Event()

    async def publish(job: IngestJob) -> None:
        sent.append(job)
        if len(sent) == 1:
            stop.set()

    task = asyncio.create_task(
        run_maintenance(
            registry,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            publish,
            interval_s=0.01,
            republish_after_s=0,
            stop=stop,
        )
    )
    await asyncio.wait_for(task, 5)

    assert sent == [registry.job]
    assert registry.statuses == [(registry.job.job_id, JobStatus.QUEUED)]
    assert registry.passes == 2
