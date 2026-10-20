"""Worker topology against a real RabbitMQ.

Run with the Compose broker up:
    RAGQA_TEST_RABBITMQ_URL=amqp://ragqa:<password>@localhost:5672/ uv run pytest -m integration
"""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import aio_pika
import pytest
from aio_pika.abc import AbstractChannel

from ragqa.domain.jobs import IngestJob
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.worker.consumer import IngestConsumer, JobHandler
from ragqa.worker.publisher import publish_job
from ragqa.worker.topology import ATTEMPT_HEADER, DeclaredTopology, Topology, declare_topology

URL = os.environ.get("RAGQA_TEST_RABBITMQ_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(URL is None, reason="RAGQA_TEST_RABBITMQ_URL not set"),
]


@pytest.fixture
async def channel() -> AsyncIterator[AbstractChannel]:
    assert URL is not None
    connection = await aio_pika.connect_robust(URL)
    async with connection:
        ch = await connection.channel(publisher_confirms=True)
        await ch.set_qos(prefetch_count=1)
        yield ch


async def _setup(channel: AbstractChannel) -> tuple[Topology, DeclaredTopology]:
    # Unique names and short delays, so runs do not interfere and finish quickly.
    topology = Topology(prefix=f"test.{uuid.uuid4().hex[:8]}", retry_delays_s=(0.2, 0.2))
    return topology, await declare_topology(channel, topology)


async def _teardown(channel: AbstractChannel, topology: Topology) -> None:
    names = [topology.jobs_queue, topology.dead_queue]
    names += [topology.retry_queue(i + 1) for i in range(topology.max_retries)]
    for name in names:
        await channel.queue_delete(name)
    for name in (topology.exchange, topology.retry_exchange, topology.dead_exchange):
        await channel.exchange_delete(name)


async def _consume(
    channel: AbstractChannel, topology: Topology, declared: DeclaredTopology, handler: JobHandler
) -> str:
    consumer = IngestConsumer(
        handler,
        topology,
        retry_exchange=declared.retry_exchange,
        dead_exchange=declared.dead_exchange,
    )
    return await declared.jobs_queue.consume(consumer)


async def _get_dead(
    declared: DeclaredTopology, timeout_s: float = 10.0
) -> aio_pika.IncomingMessage:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        message = await declared.dead_queue.get(fail=False)
        if message is not None:
            await message.ack()
            return message  # type: ignore[return-value]
        await asyncio.sleep(0.1)
    raise AssertionError("nothing reached the dead queue")


def _job() -> IngestJob:
    return IngestJob(job_id=uuid.uuid4(), document_id=uuid.uuid4(), version=1, index="idx")


async def test_transient_failures_retry_then_dead_letter(channel: AbstractChannel) -> None:
    topology, declared = await _setup(channel)
    attempts: list[IngestJob] = []

    async def always_fails(job: IngestJob) -> None:
        attempts.append(job)
        raise ConnectionError("embedder down")

    try:
        await _consume(channel, topology, declared, always_fails)
        job = _job()
        await publish_job(declared.exchange, job)

        dead = await _get_dead(declared)

        assert IngestJob.model_validate_json(dead.body) == job
        assert dead.headers[ATTEMPT_HEADER] == topology.max_retries
        assert len(attempts) == topology.max_retries + 1
    finally:
        await _teardown(channel, topology)


async def test_success_and_permanent_failure_do_not_dead_letter(channel: AbstractChannel) -> None:
    topology, declared = await _setup(channel)
    done = asyncio.Event()
    seen: list[IngestJob] = []

    async def handler(job: IngestJob) -> None:
        seen.append(job)
        if len(seen) == 2:
            done.set()
            raise PermanentIngestionError("empty file")

    try:
        await _consume(channel, topology, declared, handler)
        await publish_job(declared.exchange, _job())
        await publish_job(declared.exchange, _job())
        await asyncio.wait_for(done.wait(), 5)
        await asyncio.sleep(0.5)

        assert len(seen) == 2
        assert await declared.dead_queue.get(fail=False) is None
    finally:
        await _teardown(channel, topology)


async def test_malformed_message_is_dead_lettered_by_broker(channel: AbstractChannel) -> None:
    topology, declared = await _setup(channel)

    async def handler(job: IngestJob) -> None:
        raise AssertionError("must not be called")

    try:
        await _consume(channel, topology, declared, handler)
        await declared.exchange.publish(aio_pika.Message(b"not json"), routing_key="job")

        dead = await _get_dead(declared)

        assert dead.body == b"not json"
    finally:
        await _teardown(channel, topology)
