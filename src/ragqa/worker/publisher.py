"""Publishing ingestion jobs. Used by the API, the worker's republisher, tests and manual checks."""

import asyncio
from collections.abc import Awaitable, Callable

import aio_pika
from aio_pika.abc import AbstractExchange, AbstractRobustConnection

from ragqa.domain.jobs import IngestJob
from ragqa.observability.tracing import inject_trace_headers
from ragqa.worker.topology import JOB_ROUTING_KEY, Topology, declare_topology


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


class JobPublisher:
    """Publishes from the API process: one channel with publisher confirms, opened on first
    use together with the topology, so the API can start before the broker."""

    def __init__(
        self, connect: Callable[[], Awaitable[AbstractRobustConnection]], topology: Topology
    ) -> None:
        self._connect = connect
        self._topology = topology
        self._exchange: AbstractExchange | None = None
        self._lock = asyncio.Lock()

    async def _get_exchange(self) -> AbstractExchange:
        async with self._lock:
            if self._exchange is None or self._exchange_channel_closed():
                connection = await self._connect()
                channel = await connection.channel(publisher_confirms=True)
                self._exchange = (await declare_topology(channel, self._topology)).exchange
            return self._exchange

    def _exchange_channel_closed(self) -> bool:
        channel = getattr(self._exchange, "channel", None)
        return channel is None or channel.is_closed

    async def publish(self, job: IngestJob) -> None:
        await publish_job(await self._get_exchange(), job)
