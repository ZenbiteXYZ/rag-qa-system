"""Consumer for ingestion jobs: decode, run the handler, then ack, retry or dead-letter.

The message is acked only after its outcome is settled: handler success, permanent
failure, or a confirmed publish to the retry or dead queue. A crash before that leaves
the message unacked and RabbitMQ redelivers it, so the handler must be idempotent.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import aio_pika
import mlflow
from aio_pika.abc import AbstractExchange, AbstractIncomingMessage
from mlflow.entities import SpanType
from pydantic import ValidationError

from ragqa.domain.jobs import IngestJob
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.observability import context
from ragqa.observability.logging import get_logger, redact_text
from ragqa.observability.metrics import JobOutcome, record_job
from ragqa.observability.tracing import link_from_headers
from ragqa.worker.topology import ATTEMPT_HEADER, JOB_ROUTING_KEY, LAST_ERROR_HEADER, Topology

log = get_logger(__name__)

JobHandler = Callable[[IngestJob], Awaitable[None]]

_MAX_ERROR_HEADER_LEN = 500


def attempt_of(headers: Mapping[str, Any] | None) -> int:
    """Retry attempt from the message header; 0 for the first delivery."""
    value = (headers or {}).get(ATTEMPT_HEADER, 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class IngestConsumer:
    def __init__(
        self,
        handler: JobHandler,
        topology: Topology,
        *,
        retry_exchange: AbstractExchange,
        dead_exchange: AbstractExchange,
    ) -> None:
        self._handler = handler
        self._topology = topology
        self._retry_exchange = retry_exchange
        self._dead_exchange = dead_exchange
        self._idle = asyncio.Event()
        self._idle.set()

    async def wait_idle(self) -> None:
        """Wait until no delivery is being processed (used on shutdown)."""
        await self._idle.wait()

    async def __call__(self, message: AbstractIncomingMessage) -> None:
        self._idle.clear()
        try:
            await self._process(message)
        except Exception:
            # Ack or publish failed, usually a lost connection. Hand the message back;
            # if the channel is gone, the broker redelivers it anyway.
            log.exception("job_delivery_error")
            try:
                await message.nack(requeue=True)
            except Exception:
                log.warning("job_nack_failed")
        finally:
            context.clear()
            self._idle.set()

    async def _process(self, message: AbstractIncomingMessage) -> None:
        start = time.perf_counter()
        attempt = attempt_of(message.headers)
        try:
            job = IngestJob.model_validate_json(message.body)
        except ValidationError as exc:
            context.clear()
            log.error(
                "job_malformed",
                message_id=message.message_id,
                error=str(exc)[:_MAX_ERROR_HEADER_LEN],
            )
            # The jobs queue dead-letters rejected messages to the dead queue.
            await message.reject(requeue=False)
            record_job("malformed", attempt=attempt, duration_s=time.perf_counter() - start)
            return

        context.bind_job(
            str(job.job_id), document_id=str(job.document_id), version=job.version, attempt=attempt
        )
        link = link_from_headers(message.headers)
        attributes = {"ragqa.job.attempt": attempt, "ragqa.job.redelivered": message.redelivered}
        with mlflow.start_span(
            "ingest_job", SpanType.CHAIN, attributes, links=[link] if link else None
        ) as span:
            span.set_inputs(job.model_dump(mode="json"))
            outcome = await self._run(job, message, attempt)
            span.set_outputs({"outcome": outcome})
        record_job(outcome, attempt=attempt, duration_s=time.perf_counter() - start)

    async def _run(
        self, job: IngestJob, message: AbstractIncomingMessage, attempt: int
    ) -> JobOutcome:
        log.info("job_started", index_id=job.index_id, redelivered=message.redelivered)
        try:
            await self._handler(job)
        except PermanentIngestionError as exc:
            # TODO(slice 2): mark the job failed in Postgres with the error.
            log.error("job_failed", error_type=type(exc).__name__, error=str(exc))
            await message.ack()
            return "failed"
        except Exception as exc:
            return await self._retry_or_dead(message, attempt, exc)
        # TODO(slice 2): job status `ready` is set by the ingestion function's transaction.
        await message.ack()
        log.info("job_done")
        return "ready"

    async def _retry_or_dead(
        self, message: AbstractIncomingMessage, attempt: int, exc: Exception
    ) -> JobOutcome:
        error = redact_text(f"{type(exc).__name__}: {exc}")[:_MAX_ERROR_HEADER_LEN]
        if attempt < self._topology.max_retries:
            next_attempt = attempt + 1
            await self._republish(
                message,
                self._retry_exchange,
                self._topology.retry_routing_key(next_attempt),
                next_attempt,
                error,
            )
            await message.ack()
            log.warning(
                "job_retry_scheduled",
                next_attempt=next_attempt,
                delay_s=self._topology.retry_delays_s[attempt],
                exc_info=exc,
            )
            return "retry"

        # TODO(slice 2): mark the job dead in Postgres with the error.
        await self._republish(message, self._dead_exchange, JOB_ROUTING_KEY, attempt, error)
        await message.ack()
        log.error("job_dead", retries=attempt, exc_info=exc)
        return "dead"

    @staticmethod
    async def _republish(
        message: AbstractIncomingMessage,
        exchange: AbstractExchange,
        routing_key: str,
        attempt: int,
        error: str,
    ) -> None:
        # Keep the original headers, including traceparent, so the link still points to
        # the publisher of the first attempt.
        headers = dict(message.headers or {})
        headers[ATTEMPT_HEADER] = attempt
        headers[LAST_ERROR_HEADER] = error
        copy = aio_pika.Message(
            message.body,
            headers=headers,
            content_type=message.content_type,
            message_id=message.message_id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        # The channel uses publisher confirms, so this returns once the broker has it.
        await exchange.publish(copy, routing_key=routing_key)
