"""RabbitMQ topology for ingestion jobs (ADR-0011).

    publish --job--> [prefix]  (direct) --> [prefix].jobs  (quorum queue, consumed)
                                                 | reject / delivery limit
                                                 v
    [prefix].retry (direct) --retry.N--> [prefix].retry.N  (TTL = delay N)
                                                 | expired, dead-lettered back
                                                 v
                                          [prefix] --job--> [prefix].jobs
    [prefix].dead  (direct) --job--> [prefix].dead  (quorum queue, inspected by hand)

Retries are explicit: the worker republishes a failed job to the retry queue for the
next attempt, with the attempt number in a header, and acks the original. After the
last retry it publishes to the dead queue. Messages the worker cannot decode are
rejected and reach the dead queue through the jobs queue's dead-letter exchange.

Queue arguments cannot be changed on an existing queue (RabbitMQ answers with
PRECONDITION_FAILED). Changing delays or queue types means deleting the queues first.
"""

from dataclasses import dataclass

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractQueue

from ragqa.settings import Settings

JOB_ROUTING_KEY = "job"
ATTEMPT_HEADER = "x-ragqa-attempt"
LAST_ERROR_HEADER = "x-ragqa-last-error"

# Redeliveries without an ack (worker crash mid-job) before the broker dead-letters
# the message. Protects against a job that crashes the process every time.
DELIVERY_LIMIT = 5


@dataclass(frozen=True)
class Topology:
    prefix: str = "ragqa.ingest"
    retry_delays_s: tuple[float, ...] = (5.0, 10.0, 20.0)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Topology":
        base = settings.ingest_retry_base_delay_s
        delays = tuple(base * 2**i for i in range(settings.ingest_max_retries))
        return cls(retry_delays_s=delays)

    @property
    def max_retries(self) -> int:
        return len(self.retry_delays_s)

    @property
    def exchange(self) -> str:
        return self.prefix

    @property
    def jobs_queue(self) -> str:
        return f"{self.prefix}.jobs"

    @property
    def retry_exchange(self) -> str:
        return f"{self.prefix}.retry"

    def retry_queue(self, attempt: int) -> str:
        return f"{self.prefix}.retry.{attempt}"

    @staticmethod
    def retry_routing_key(attempt: int) -> str:
        return f"retry.{attempt}"

    @property
    def dead_exchange(self) -> str:
        return f"{self.prefix}.dead"

    @property
    def dead_queue(self) -> str:
        return f"{self.prefix}.dead"


@dataclass(frozen=True)
class DeclaredTopology:
    exchange: AbstractExchange
    retry_exchange: AbstractExchange
    dead_exchange: AbstractExchange
    jobs_queue: AbstractQueue
    dead_queue: AbstractQueue


async def declare_topology(channel: AbstractChannel, topology: Topology) -> DeclaredTopology:
    """Declare exchanges and queues. Idempotent while the arguments stay the same."""
    direct = aio_pika.ExchangeType.DIRECT
    exchange = await channel.declare_exchange(topology.exchange, direct, durable=True)
    retry_exchange = await channel.declare_exchange(topology.retry_exchange, direct, durable=True)
    dead_exchange = await channel.declare_exchange(topology.dead_exchange, direct, durable=True)

    dead_queue = await channel.declare_queue(
        topology.dead_queue, durable=True, arguments={"x-queue-type": "quorum"}
    )
    await dead_queue.bind(dead_exchange, JOB_ROUTING_KEY)

    jobs_queue = await channel.declare_queue(
        topology.jobs_queue,
        durable=True,
        arguments={
            "x-queue-type": "quorum",
            "x-delivery-limit": DELIVERY_LIMIT,
            "x-dead-letter-exchange": topology.dead_exchange,
            "x-dead-letter-routing-key": JOB_ROUTING_KEY,
        },
    )
    await jobs_queue.bind(exchange, JOB_ROUTING_KEY)

    for attempt, delay_s in enumerate(topology.retry_delays_s, start=1):
        retry_queue = await channel.declare_queue(
            topology.retry_queue(attempt),
            durable=True,
            arguments={
                "x-message-ttl": int(delay_s * 1000),
                "x-dead-letter-exchange": topology.exchange,
                "x-dead-letter-routing-key": JOB_ROUTING_KEY,
            },
        )
        await retry_queue.bind(retry_exchange, topology.retry_routing_key(attempt))

    return DeclaredTopology(exchange, retry_exchange, dead_exchange, jobs_queue, dead_queue)
