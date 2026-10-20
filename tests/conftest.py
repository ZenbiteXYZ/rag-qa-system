import io
import json
from collections.abc import Callable, Iterator

import mlflow
import pytest

from ragqa.observability import context
from ragqa.observability.logging import configure_logging
from ragqa.settings import Settings

# Unit tests never export traces. mlflow.start_span returns no-op spans.
mlflow.tracing.disable()


@pytest.fixture
def settings() -> Settings:
    # Dummy credentials: unit tests never connect to these.
    return Settings(
        postgres_dsn="postgresql+asyncpg://ragqa:test-password@localhost:1/ragqa",
        rabbitmq_url="amqp://ragqa:test-password@localhost:1/",
        service_name="ragqa-test",
        env="test",
        log_level="DEBUG",
    )


LogLines = Callable[[], list[dict[str, object]]]


def log_buffer() -> Iterator[LogLines]:
    """Configure JSON logging into a buffer and yield a reader of the parsed lines."""
    buffer = io.StringIO()
    configure_logging(service="ragqa-test", env="test", level="DEBUG", stream=buffer)
    context.clear()

    def read() -> list[dict[str, object]]:
        return [json.loads(line) for line in buffer.getvalue().splitlines() if line]

    yield read
    context.clear()


@pytest.fixture
def log_lines() -> Iterator[LogLines]:
    yield from log_buffer()
