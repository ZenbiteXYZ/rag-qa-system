"""Dependency checks behind `GET /ready`.

Required dependencies decide readiness. Optional ones (embedder, MLflow) are reported
but do not take the API out of rotation.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from opentelemetry.instrumentation.utils import suppress_instrumentation
from pydantic import BaseModel
from sqlalchemy import text

from ragqa.api.resources import Resources
from ragqa.observability.logging import get_logger, redact_text
from ragqa.observability.metrics import Dependency, record_dependency_error
from ragqa.settings import Settings

log = get_logger(__name__)

Probe = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class Check:
    name: Dependency
    required: bool
    probe: Probe


class CheckResult(BaseModel):
    status: Literal["ok", "fail"]
    required: bool
    latency_ms: float
    error: str | None = None


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckResult]


async def run_checks(checks: list[Check], timeout_s: float = 2.0) -> ReadyResponse:
    # No spans for probes: they run every few seconds and would flood MLflow.
    with suppress_instrumentation():
        results = await asyncio.gather(*(_run_one(c, timeout_s) for c in checks))
    by_name: dict[str, CheckResult] = {c.name: r for c, r in zip(checks, results, strict=True)}
    ready = all(r.status == "ok" for r in results if r.required)
    return ReadyResponse(status="ready" if ready else "not_ready", checks=by_name)


async def _run_one(check: Check, timeout_s: float) -> CheckResult:
    start = time.perf_counter()
    error: str | None = None
    try:
        await asyncio.wait_for(check.probe(), timeout_s)
    except TimeoutError:
        error = f"timeout after {timeout_s}s"
    except Exception as exc:
        error = redact_text(f"{type(exc).__name__}: {exc}")[:300]
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    if error is not None:
        record_dependency_error(check.name)
        log.warning("dependency_check_failed", dependency=check.name, error=error)
    return CheckResult(
        status="ok" if error is None else "fail",
        required=check.required,
        latency_ms=latency_ms,
        error=error,
    )


def build_checks(settings: Settings, resources: Resources) -> list[Check]:
    async def postgres() -> None:
        async with resources.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def rabbitmq() -> None:
        connection = await resources.amqp.get()
        channel = await connection.channel()
        await channel.close()

    def http_probe(url: str) -> Probe:
        async def probe() -> None:
            response = await resources.http.get(url)
            response.raise_for_status()

        return probe

    return [
        Check("postgres", True, postgres),
        Check("qdrant", True, http_probe(f"{settings.qdrant_url}/readyz")),
        Check("rabbitmq", True, rabbitmq),
        Check("embedder", False, http_probe(f"{settings.embedder_url}/health")),
        Check("mlflow", False, http_probe(f"{settings.mlflow_tracking_uri}/health")),
    ]
