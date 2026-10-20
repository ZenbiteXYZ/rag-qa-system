"""Structured logging for API and worker.

structlog renders every record as one JSON line. Records from stdlib loggers (uvicorn,
aio-pika, httpx) go through the same processors via `ProcessorFormatter`, so the output
has one shape. Each line carries the bound context (request_id, job_id, ...), the
service and env, and the current OTel trace and span ids.

Usage everywhere:

    from ragqa.observability.logging import get_logger

    log = get_logger(__name__)
    log.info("version_ready", document_id=doc_id, version=3)
"""

import logging
import re
import sys
from collections.abc import Mapping
from typing import Any, TextIO

import structlog
from opentelemetry import trace
from pydantic import SecretStr
from structlog.typing import EventDict, Processor, WrappedLogger

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "x_api_key",
    }
)
# Suffix match catches `postgres_password`, `litellm_api_key`, ... but not `max_tokens`.
_SENSITIVE_SUFFIXES = ("_password", "_secret", "_token", "_api_key", "_apikey")

_PATTERNS = [
    # Password in a URL or DSN: scheme://user:password@host
    (
        re.compile(r"(?P<head>\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^:/@\s]*:)[^@\s]+(?=@)"),
        rf"\g<head>{REDACTED}",
    ),
    # OpenAI-style keys: sk-..., sk-proj-..., sk-or-v1-...
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), f"sk-{REDACTED}"),
    # Authorization header values
    (re.compile(r"(?i)(?P<head>\bbearer\s+)[A-Za-z0-9._~+/\-]+=*"), rf"\g<head>{REDACTED}"),
]

# Third-party loggers that are noisy at INFO.
_QUIET_LOGGERS = ("httpx", "httpcore", "aiormq", "aio_pika", "urllib3")


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def redact_text(text: str) -> str:
    """Mask secrets that appear inside free text (messages, tracebacks, error strings)."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_value(value: Any) -> Any:
    """Recursively mask secrets in a value. Mapping keys with sensitive names are masked."""
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            k: REDACTED if isinstance(k, str) and _is_sensitive_key(k) else redact_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(redact_value(v) for v in value)
    return value


def _redact_processor(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    for key, value in event_dict.items():
        if key.startswith("_"):  # structlog internals such as _record
            continue
        event_dict[key] = REDACTED if _is_sensitive_key(key) else redact_value(value)
    return event_dict


def _add_otel_ids(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def _service_fields(service: str, env: str) -> Processor:
    def add(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("service", service)
        event_dict.setdefault("env", env)
        return event_dict

    return add


def _shared_processors(service: str, env: str) -> list[Processor]:
    """Processors applied to both structlog and stdlib records, before rendering."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _service_fields(service, env),
        _add_otel_ids,
        structlog.processors.StackInfoRenderer(),
        # Format tracebacks before redaction, so secrets inside them are masked too.
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        _redact_processor,
    ]


def configure_logging(
    *,
    service: str,
    env: str,
    level: str = "INFO",
    fmt: str = "json",
    stream: TextIO | None = None,
) -> None:
    """Route structlog and stdlib logging to stdout (or `stream`) through one formatter.

    Safe to call again: the previous handler is replaced.
    """
    shared = _shared_processors(service, env)
    stream = stream or sys.stdout
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=stream.isatty())
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Off so that reconfiguring (tests, worker vs API) takes effect on existing loggers.
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn and MLflow install their own plain-text handlers (MLflow on import). Send their
    # records to the root handler instead. The access log is replaced by our middleware.
    for name in ("uvicorn", "uvicorn.error", "mlflow"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = []
        uv_logger.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
    access.disabled = True

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, logging.getLevelName(level)))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
