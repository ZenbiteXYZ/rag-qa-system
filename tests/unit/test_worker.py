import uuid
from typing import Any

import pytest
from prometheus_client import REGISTRY

from ragqa.domain.jobs import IngestJob
from ragqa.ingestion.errors import PermanentIngestionError, TransientIngestionError
from ragqa.settings import Settings
from ragqa.worker.consumer import IngestConsumer, attempt_of
from ragqa.worker.topology import ATTEMPT_HEADER, LAST_ERROR_HEADER, Topology
from tests.conftest import LogLines

TOPOLOGY = Topology(prefix="test.ingest", retry_delays_s=(1.0, 2.0, 4.0))


def _job() -> IngestJob:
    return IngestJob(
        job_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        version=1,
        index_id=1,
    )


class FakeMessage:
    def __init__(self, body: bytes, headers: dict[str, Any] | None = None) -> None:
        self.body = body
        self.headers = headers or {}
        self.message_id = "msg-1"
        self.content_type = "application/json"
        self.redelivered = False
        self.acked = False
        self.rejected: bool | None = None
        self.nacked: bool | None = None

    async def ack(self) -> None:
        self.acked = True

    async def reject(self, requeue: bool = False) -> None:
        self.rejected = requeue

    async def nack(self, requeue: bool = True) -> None:
        self.nacked = requeue


class FakeExchange:
    def __init__(self, fail: bool = False) -> None:
        self.published: list[tuple[str, Any]] = []
        self.fail = fail

    async def publish(self, message: Any, routing_key: str) -> None:
        if self.fail:
            raise ConnectionError("channel closed")
        self.published.append((routing_key, message))


def _consumer(handler: Any, retry: FakeExchange, dead: FakeExchange) -> IngestConsumer:
    return IngestConsumer(handler, TOPOLOGY, retry_exchange=retry, dead_exchange=dead)  # type: ignore[arg-type]


def _jobs(outcome: str) -> float:
    return REGISTRY.get_sample_value("ragqa_ingest_jobs_total", {"outcome": outcome}) or 0.0


async def test_success_acks_without_publishing() -> None:
    seen: list[IngestJob] = []

    async def handler(job: IngestJob) -> None:
        seen.append(job)

    job = _job()
    message, retry, dead = (
        FakeMessage(job.model_dump_json().encode()),
        FakeExchange(),
        FakeExchange(),
    )
    before = _jobs("ready")

    await _consumer(handler, retry, dead)(message)  # type: ignore[arg-type]

    assert seen == [job]
    assert message.acked
    assert retry.published == dead.published == []
    assert _jobs("ready") == before + 1


async def test_permanent_error_acks_without_retry() -> None:
    async def handler(job: IngestJob) -> None:
        raise PermanentIngestionError("empty file")

    message, retry, dead = (
        FakeMessage(_job().model_dump_json().encode()),
        FakeExchange(),
        FakeExchange(),
    )

    await _consumer(handler, retry, dead)(message)  # type: ignore[arg-type]

    assert message.acked
    assert retry.published == dead.published == []


async def test_transient_error_schedules_next_retry(log_lines: LogLines) -> None:
    async def handler(job: IngestJob) -> None:
        raise TransientIngestionError("embedder down at http://u:tei-pass@tei:80/")

    headers = {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}
    message = FakeMessage(_job().model_dump_json().encode(), headers)
    retry, dead = FakeExchange(), FakeExchange()

    await _consumer(handler, retry, dead)(message)  # type: ignore[arg-type]

    assert message.acked
    assert dead.published == []
    ((routing_key, copy),) = retry.published
    assert routing_key == "retry.1"
    assert copy.headers[ATTEMPT_HEADER] == 1
    assert copy.headers["traceparent"] == headers["traceparent"]
    assert "tei-pass" not in copy.headers[LAST_ERROR_HEADER]
    assert copy.body == message.body
    (line,) = [x for x in log_lines() if x["event"] == "job_retry_scheduled"]
    assert line["attempt"] == 0
    assert line["delay_s"] == 1.0


async def test_last_retry_goes_to_dead_queue() -> None:
    async def handler(job: IngestJob) -> None:
        raise RuntimeError("still failing")

    message = FakeMessage(_job().model_dump_json().encode(), {ATTEMPT_HEADER: 3})
    retry, dead = FakeExchange(), FakeExchange()

    await _consumer(handler, retry, dead)(message)  # type: ignore[arg-type]

    assert message.acked
    assert retry.published == []
    ((routing_key, copy),) = dead.published
    assert routing_key == "job"
    assert copy.headers[ATTEMPT_HEADER] == 3


async def test_malformed_message_is_rejected_to_dead_letter() -> None:
    called = False

    async def handler(job: IngestJob) -> None:
        nonlocal called
        called = True

    message = FakeMessage(b'{"not": "a job"}')

    await _consumer(handler, FakeExchange(), FakeExchange())(message)  # type: ignore[arg-type]

    assert not called
    assert message.rejected is False
    assert not message.acked


async def test_failed_republish_hands_message_back() -> None:
    async def handler(job: IngestJob) -> None:
        raise RuntimeError("transient")

    message = FakeMessage(_job().model_dump_json().encode())

    await _consumer(handler, FakeExchange(fail=True), FakeExchange())(message)  # type: ignore[arg-type]

    assert not message.acked
    assert message.nacked is True


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (None, 0),
        ({}, 0),
        ({ATTEMPT_HEADER: 2}, 2),
        ({ATTEMPT_HEADER: "x"}, 0),
        ({ATTEMPT_HEADER: -1}, 0),
    ],
)
def test_attempt_of(headers: dict[str, Any] | None, expected: int) -> None:
    assert attempt_of(headers) == expected


def test_topology_from_settings(settings: Settings) -> None:
    topology = Topology.from_settings(settings)

    assert topology.retry_delays_s == (5.0, 10.0, 20.0)
    assert topology.max_retries == 3
    assert topology.jobs_queue == "ragqa.ingest.jobs"
    assert topology.retry_queue(2) == "ragqa.ingest.retry.2"
    assert topology.dead_queue == "ragqa.ingest.dead"
