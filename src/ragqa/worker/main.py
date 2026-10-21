"""Worker process: consume ingestion jobs until SIGTERM. Run with `python -m ragqa.worker`."""

import asyncio
import signal

import aio_pika
from prometheus_client import start_http_server

from ragqa.embeddings.tokens import hf_token_counter_factory
from ragqa.ingestion.service import IngestionService
from ragqa.observability.logging import configure_logging, get_logger
from ragqa.observability.metrics import WORKER_CONSUMING
from ragqa.observability.tracing import setup_tracing, shutdown_tracing
from ragqa.settings import Settings, get_settings
from ragqa.wiring import Backends
from ragqa.worker.consumer import IngestConsumer
from ragqa.worker.maintenance import run_maintenance
from ragqa.worker.publisher import publish_job
from ragqa.worker.reporter import RegistryJobReporter
from ragqa.worker.topology import Topology, declare_topology

log = get_logger(__name__)


async def run(settings: Settings, backends: Backends, stop: asyncio.Event) -> None:
    topology = Topology.from_settings(settings)
    connection = await aio_pika.connect_robust(
        settings.rabbitmq_url, client_properties={"connection_name": settings.service_name}
    )
    async with connection:
        channel = await connection.channel(publisher_confirms=True)
        # One unacked job at a time (ADR-0011).
        await channel.set_qos(prefetch_count=1)
        declared = await declare_topology(channel, topology)
        service = IngestionService(
            registry=backends.registry,
            files=backends.files,
            chunk_store=backends.chunk_store,
            embedder=backends.embedder,
            parsers=backends.parsers,
            chunkers=backends.chunkers,
            token_counter=backends.token_counter,
        )
        consumer = IngestConsumer(
            service.ingest,
            topology,
            retry_exchange=declared.retry_exchange,
            dead_exchange=declared.dead_exchange,
            reporter=RegistryJobReporter(backends.registry),
        )
        tag = await declared.jobs_queue.consume(consumer)
        maintenance = asyncio.create_task(
            run_maintenance(
                backends.registry,
                backends.files,
                backends.chunk_store,
                lambda job: publish_job(declared.exchange, job),
                interval_s=settings.maintenance_interval_s,
                republish_after_s=settings.republish_after_s,
                stop=stop,
            )
        )
        WORKER_CONSUMING.set(1)
        log.info(
            "worker_started", queue=topology.jobs_queue, retry_delays_s=topology.retry_delays_s
        )

        await stop.wait()

        log.info("worker_stopping")
        await declared.jobs_queue.cancel(tag)
        await maintenance
        WORKER_CONSUMING.set(0)
        try:
            await asyncio.wait_for(consumer.wait_idle(), settings.worker_shutdown_grace_s)
        except TimeoutError:
            # The unacked message is redelivered after the connection closes.
            log.warning("worker_shutdown_timeout", grace_s=settings.worker_shutdown_grace_s)
    log.info("worker_stopped")


async def _main(settings: Settings) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    backends = Backends.create(settings, token_counter=hf_token_counter_factory(settings))
    try:
        await run(settings, backends, stop)
    finally:
        await backends.close()


def main() -> None:
    settings = get_settings()
    if "service_name" not in settings.model_fields_set:
        settings = settings.model_copy(update={"service_name": "ragqa-worker"})
    configure_logging(
        service=settings.service_name,
        env=settings.env,
        level=settings.log_level,
        fmt=settings.log_format,
    )
    start_http_server(settings.worker_metrics_port)
    setup_tracing(settings)
    try:
        asyncio.run(_main(settings))
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    main()
