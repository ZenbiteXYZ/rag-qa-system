import io
import json
import logging

from opentelemetry.sdk.trace import TracerProvider
from pydantic import SecretStr

from ragqa.observability import context
from ragqa.observability.logging import REDACTED, get_logger, redact_text
from tests.conftest import LogLines, log_buffer

log = get_logger("tests.logging")


def test_structlog_line_has_expected_fields(log_lines: LogLines) -> None:
    log.info("something_happened", count=3)

    (line,) = log_lines()
    assert line["event"] == "something_happened"
    assert line["count"] == 3
    assert line["level"] == "info"
    assert line["logger"] == "tests.logging"
    assert line["service"] == "ragqa-test"
    assert line["env"] == "test"
    assert str(line["timestamp"]).endswith("Z")
    assert "trace_id" not in line  # no active span


def test_stdlib_logger_is_rendered_as_json_with_context(log_lines: LogLines) -> None:
    context.bind_request("req-1")
    logging.getLogger("uvicorn.error").warning("stdlib says %s", "hi")

    (line,) = log_lines()
    assert line["event"] == "stdlib says hi"
    assert line["level"] == "warning"
    assert line["request_id"] == "req-1"
    assert line["service"] == "ragqa-test"


def test_mlflow_logger_is_rendered_as_json() -> None:
    # MLflow attaches its own plain-text handler and stops propagation on import.
    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logger.handlers = [logging.StreamHandler(io.StringIO())]
    mlflow_logger.propagate = False

    reader = log_buffer()
    log_lines = next(reader)
    logging.getLogger("mlflow.tracking.fluent").warning("Creating a new experiment.")

    (line,) = log_lines()
    assert line["event"] == "Creating a new experiment."
    assert line["logger"] == "mlflow.tracking.fluent"
    reader.close()


def test_each_line_is_a_single_json_object(log_lines: LogLines) -> None:
    try:
        raise ValueError("multi\nline")
    except ValueError:
        log.exception("failed")

    (line,) = log_lines()
    assert "ValueError: multi" in str(line["exception"])


def test_bind_request_clears_previous_context(log_lines: LogLines) -> None:
    context.bind_job("job-1", document_id="doc-1", version=2, attempt=1)
    log.info("in_job")
    context.bind_request("req-2")
    log.info("in_request")

    job_line, request_line = log_lines()
    assert job_line["job_id"] == "job-1"
    assert job_line["version"] == 2
    assert job_line["attempt"] == 1
    assert request_line["request_id"] == "req-2"
    assert "job_id" not in request_line


def test_bound_restores_previous_values(log_lines: LogLines) -> None:
    context.bind_request("req-3")
    with context.bound(stage="rewrite"):
        log.info("inside")
    log.info("outside")

    inside, outside = log_lines()
    assert inside["stage"] == "rewrite"
    assert "stage" not in outside
    assert outside["request_id"] == "req-3"


def test_trace_and_span_ids_come_from_current_span(log_lines: LogLines) -> None:
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("op") as span:
        log.info("traced")
        ctx = span.get_span_context()

    (line,) = log_lines()
    assert line["trace_id"] == format(ctx.trace_id, "032x")
    assert line["span_id"] == format(ctx.span_id, "016x")


def test_sensitive_keys_are_redacted(log_lines: LogLines) -> None:
    log.info(
        "config",
        password="hunter2",
        litellm_api_key="abc",
        Authorization="Bearer xyz",
        nested={"db": {"postgres_password": "p", "host": "db"}},
        secret_obj=SecretStr("s3cret"),
        max_tokens=512,
        prompt_tokens=10,
    )

    (line,) = log_lines()
    raw = json.dumps(line)
    for secret in ("hunter2", '"abc"', "xyz", "s3cret"):
        assert secret not in raw
    assert line["password"] == REDACTED
    assert line["nested"] == {"db": {"postgres_password": REDACTED, "host": "db"}}
    assert line["secret_obj"] == REDACTED
    assert line["max_tokens"] == 512
    assert line["prompt_tokens"] == 10


def test_secrets_inside_text_are_redacted(log_lines: LogLines) -> None:
    dsn = "postgresql+asyncpg://ragqa:pg-pass-123@postgres:5432/ragqa"
    log.info(f"connecting to {dsn}", url="amqp://ragqa:mq-pass@rabbitmq:5672/")
    try:
        raise ConnectionError(f"cannot reach {dsn} with sk-or-v1-abcdef1234567890")
    except ConnectionError:
        log.exception("connect_failed")

    raw = "\n".join(json.dumps(line) for line in log_lines())
    assert "pg-pass-123" not in raw
    assert "mq-pass" not in raw
    assert "abcdef1234567890" not in raw
    assert "postgresql+asyncpg://ragqa:[REDACTED]@postgres:5432/ragqa" in raw


def test_redact_text_cases() -> None:
    assert redact_text("amqp://u:p@h/") == f"amqp://u:{REDACTED}@h/"
    assert redact_text("key sk-proj-AAAAAAAAAAAA end") == f"key sk-{REDACTED} end"
    assert redact_text("Authorization: Bearer abc.def") == f"Authorization: Bearer {REDACTED}"
    # Plain URLs and short sk- words are left alone.
    assert redact_text("http://qdrant:6333/readyz") == "http://qdrant:6333/readyz"
    assert redact_text("task-sk-1") == "task-sk-1"
