"""Per-request and per-job log context, stored in contextvars.

Values bound here appear on every log line emitted in the same asyncio task, including
lines from stdlib loggers. Each request or job starts from a clean context, so ids
never leak between them.
"""

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from structlog.contextvars import bind_contextvars, bound_contextvars, clear_contextvars

# Accept caller-provided ids only in a safe shape, so they cannot inject into log lines.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")


def new_request_id() -> str:
    return uuid.uuid4().hex


def valid_request_id(value: str | None) -> str | None:
    if value is not None and _REQUEST_ID_RE.fullmatch(value):
        return value
    return None


def bind_request(request_id: str) -> None:
    """Start a fresh context for one HTTP request."""
    clear_contextvars()
    bind_contextvars(request_id=request_id)


def bind_job(
    job_id: str,
    *,
    document_id: str | None = None,
    version: int | None = None,
    attempt: int | None = None,
) -> None:
    """Start a fresh context for one ingestion job delivery."""
    clear_contextvars()
    fields: dict[str, Any] = {"job_id": job_id}
    if document_id is not None:
        fields["document_id"] = document_id
    if version is not None:
        fields["version"] = version
    if attempt is not None:
        fields["attempt"] = attempt
    bind_contextvars(**fields)


def clear() -> None:
    clear_contextvars()


@contextmanager
def bound(**fields: Any) -> Iterator[None]:
    """Bind fields for the duration of a block, then restore the previous values."""
    with bound_contextvars(**fields):
        yield
