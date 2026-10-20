"""Tracing: one OTel TracerProvider shared by MLflow and OTel auto-instrumentation.

With `MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false`, MLflow adds its span processors to the
global provider (`mlflow.tracing.set_destination`), so FastAPI and httpx spans and our
MLflow spans (`traced_stage`, `llm_call`) end up in one MLflow trace. The MLflow trace id
is `tr-` plus the OTel trace id that the logs carry as `trace_id`.

Hooks for pipeline code:

    with traced_stage("rewrite", reads=["question"], writes=["rewritten_query"]) as span:
        span.set_inputs({"question": state.question})
        ...
        span.set_outputs({"rewritten_query": result})

    with llm_call("generator") as call:
        response = await client.chat.completions.create(...)
        call.prompt_tokens = response.usage.prompt_tokens
        call.completion_tokens = response.usage.completion_tokens
"""

import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities import Link, LiveSpan, SpanType
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from ragqa.observability import context
from ragqa.observability.logging import get_logger
from ragqa.observability.metrics import (
    StageOutcome,
    record_dependency_error,
    record_llm_call,
    record_stage,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ragqa.settings import Settings

log = get_logger(__name__)

# Probe and scrape endpoints are not traced, otherwise every healthcheck becomes a trace.
UNTRACED_URLS = "health,ready,metrics"

_provider: TracerProvider | None = None


def setup_tracing(settings: "Settings", *, attempts: int = 5, delay_s: float = 2.0) -> bool:
    """Install the tracer provider and point MLflow at the trace experiment.

    Returns True when traces are exported. If MLflow cannot be reached, the service keeps
    running with trace ids in logs but no exported traces.
    """
    global _provider
    if not settings.tracing_enabled:
        mlflow.tracing.disable()
        log.info("tracing_disabled")
        return False

    # Must be set before MLflow initializes its tracer provider.
    os.environ["MLFLOW_USE_DEFAULT_TRACER_PROVIDER"] = "false"
    resource = Resource.create(
        {"service.name": settings.service_name, "deployment.environment.name": settings.env}
    )
    _provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(_provider)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = _get_experiment_id(settings.mlflow_experiment, attempts, delay_s)
    if experiment_id is None:
        return False
    mlflow.tracing.set_destination(MlflowExperimentLocation(experiment_id))

    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()
    log.info("tracing_enabled", experiment=settings.mlflow_experiment, experiment_id=experiment_id)
    return True


def _get_experiment_id(name: str, attempts: int, delay_s: float) -> str | None:
    for attempt in range(1, attempts + 1):
        try:
            return str(mlflow.set_experiment(name).experiment_id)
        except Exception as exc:
            record_dependency_error("mlflow")
            log.warning("mlflow_unavailable", attempt=attempt, error=str(exc))
            time.sleep(delay_s)
    log.error("tracing_export_off", reason="mlflow unreachable at startup")
    return None


def instrument_fastapi(app: "FastAPI") -> None:
    """Server spans for every request except probes. Spans are no-ops until a provider is set."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app, excluded_urls=UNTRACED_URLS, exclude_spans=["receive", "send"]
    )


def shutdown_tracing() -> None:
    """Flush pending traces. Call on process shutdown."""
    if _provider is None:
        return
    try:
        mlflow.flush_trace_async_logging()
    except Exception as exc:
        log.warning("trace_flush_failed", error=str(exc))
    _provider.shutdown()


# --- Context propagation over RabbitMQ (W3C traceparent in message headers) ---


def inject_trace_headers() -> dict[str, str]:
    """Headers that carry the current trace context, for a message about to be published."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def link_from_headers(headers: Mapping[str, Any] | None) -> Link | None:
    """A span link to the publisher's span, if the message carries a valid traceparent.

    The worker starts its own trace per job and links it to the publisher, because the
    publishing request has usually finished and been exported before the job runs.
    """
    if not headers:
        return None
    carrier = {
        k: v.decode() if isinstance(v, bytes) else str(v)
        for k, v in headers.items()
        if v is not None
    }
    span_ctx = trace.get_current_span(propagate.extract(carrier)).get_span_context()
    if not span_ctx.is_valid:
        return None
    # mypy sees an abstract classmethod on MLflow's base class; the dataclass is concrete.
    return Link(  # type: ignore[abstract]
        trace_id=format(span_ctx.trace_id, "032x"),
        span_id=format(span_ctx.span_id, "016x"),
        attributes={"ragqa.link": "publisher"},
    )


# --- Hooks for pipeline stages and LLM calls ---


@contextmanager
def traced_stage(
    name: str,
    *,
    reads: Sequence[str] = (),
    writes: Sequence[str] = (),
    span_type: str = SpanType.CHAIN,
) -> Iterator[LiveSpan]:
    """Span, latency metric and log context for one pipeline stage.

    `reads` and `writes` name the state fields the stage uses. They are recorded on the
    span; inputs and outputs are set by the caller with `span.set_inputs/set_outputs`.
    """
    attributes = {"ragqa.stage.reads": list(reads), "ragqa.stage.writes": list(writes)}
    start = time.perf_counter()
    outcome: StageOutcome = "ok"
    with context.bound(stage=name), mlflow.start_span(name, span_type, attributes) as span:
        try:
            yield span
        except Exception:
            outcome = "error"
            raise
        finally:
            duration = time.perf_counter() - start
            record_stage(name, outcome, duration)
            log.debug("stage_finished", outcome=outcome, duration_ms=round(duration * 1000, 1))


@dataclass
class LlmCall:
    """Filled in by the caller inside `llm_call`; recorded when the block exits."""

    span: LiveSpan
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


@contextmanager
def llm_call(model_alias: str, *, name: str | None = None) -> Iterator[LlmCall]:
    """Span, metrics and log line for one call to the LiteLLM proxy."""
    start = time.perf_counter()
    outcome: StageOutcome = "ok"
    attributes = {"ragqa.llm.model_alias": model_alias}
    with mlflow.start_span(name or f"llm:{model_alias}", SpanType.CHAT_MODEL, attributes) as span:
        call = LlmCall(span=span)
        try:
            yield call
        except Exception:
            outcome = "error"
            record_dependency_error("llm")
            raise
        finally:
            duration = time.perf_counter() - start
            record_llm_call(
                model_alias,
                outcome=outcome,
                duration_s=duration,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                cost_usd=call.cost_usd,
            )
            _set_usage(span, call)
            log.info(
                "llm_call",
                model_alias=model_alias,
                outcome=outcome,
                duration_ms=round(duration * 1000, 1),
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                cost_usd=call.cost_usd,
            )


def _set_usage(span: LiveSpan, call: LlmCall) -> None:
    if call.prompt_tokens is None and call.completion_tokens is None:
        return
    prompt, completion = call.prompt_tokens or 0, call.completion_tokens or 0
    # MLflow sums this attribute into the trace-level token usage.
    span.set_attribute(
        SpanAttributeKey.CHAT_USAGE,
        {
            TokenUsageKey.INPUT_TOKENS: prompt,
            TokenUsageKey.OUTPUT_TOKENS: completion,
            TokenUsageKey.TOTAL_TOKENS: prompt + completion,
        },
    )
    if call.cost_usd is not None:
        span.set_attribute("ragqa.llm.cost_usd", call.cost_usd)
