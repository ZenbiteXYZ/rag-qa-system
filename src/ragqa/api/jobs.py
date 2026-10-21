"""Ingestion job status (story A2)."""

import uuid

from fastapi import APIRouter

from ragqa.api.deps import ResourcesDep
from ragqa.api.schemas import JobOut

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}")
async def get_job(job_id: uuid.UUID, resources: ResourcesDep) -> JobOut:
    return JobOut.model_validate(await resources.backends.registry.get_job(job_id))
