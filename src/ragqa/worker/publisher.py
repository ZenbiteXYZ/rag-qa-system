"""Publishing ingestion jobs. Used by the API (slice 2), tests and manual checks."""

import aio_pika
from aio_pika.abc import AbstractExchange

from ragqa.domain.jobs import IngestJob
from ragqa.observability.tracing import inject_trace_headers
from ragqa.worker.topology import JOB_ROUTING_KEY


def job_message(job: IngestJob) -> aio_pika.Message:
    return aio_pika.Message(
        job.model_dump_json().encode(),
        headers=dict(inject_trace_headers()),
        content_type="application/json",
        message_id=str(job.job_id),
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    )


async def publish_job(exchange: AbstractExchange, job: IngestJob) -> None:
    await exchange.publish(job_message(job), routing_key=JOB_ROUTING_KEY)
