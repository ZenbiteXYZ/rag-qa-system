"""Prometheus metrics shared by API and worker.

Labels must stay low-cardinality: route templates, stage names, model aliases, outcomes.
Never put question text, document ids or user input into a label.
"""

from typing import Literal

from prometheus_client import Counter, Gauge, Histogram

Dependency = Literal["postgres", "qdrant", "rabbitmq", "embedder", "llm", "mlflow"]
JobOutcome = Literal["ready", "retry", "failed", "dead", "malformed"]
StageOutcome = Literal["ok", "error"]

# Route label for requests that matched no route, so random paths cannot add series.
UNMATCHED_ROUTE = "__unmatched__"

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
_JOB_BUCKETS = (0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800)

HTTP_REQUESTS = Counter(
    "ragqa_http_requests_total",
    "HTTP requests by route template and status code.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "ragqa_http_request_duration_seconds",
    "HTTP request latency by route template.",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
)

STAGE_LATENCY = Histogram(
    "ragqa_stage_duration_seconds",
    "Pipeline stage latency.",
    ["stage", "outcome"],
    buckets=_LATENCY_BUCKETS,
)

LLM_REQUESTS = Counter(
    "ragqa_llm_requests_total",
    "LLM calls through the LiteLLM proxy.",
    ["model_alias", "outcome"],
)
LLM_LATENCY = Histogram(
    "ragqa_llm_request_duration_seconds",
    "LLM call latency.",
    ["model_alias"],
    buckets=_LATENCY_BUCKETS,
)
LLM_TOKENS = Counter(
    "ragqa_llm_tokens_total",
    "LLM tokens by kind (prompt, completion).",
    ["model_alias", "kind"],
)
LLM_COST = Counter(
    "ragqa_llm_cost_usd_total",
    "LLM cost in USD as reported by the proxy.",
    ["model_alias"],
)

INGEST_JOBS = Counter(
    "ragqa_ingest_jobs_total",
    "Ingestion job deliveries by outcome.",
    ["outcome"],
)
INGEST_JOB_LATENCY = Histogram(
    "ragqa_ingest_job_duration_seconds",
    "Time spent on one ingestion job delivery.",
    ["outcome"],
    buckets=_JOB_BUCKETS,
)
INGEST_JOB_ATTEMPT = Histogram(
    "ragqa_ingest_job_attempt",
    "Attempt number (0 = first delivery) at which a job finished.",
    ["outcome"],
    buckets=(0, 1, 2, 3, 4),
)
WORKER_CONSUMING = Gauge(
    "ragqa_worker_consuming",
    "1 while the worker is subscribed to the ingestion queue.",
)

DEPENDENCY_ERRORS = Counter(
    "ragqa_dependency_errors_total",
    "Failed calls or checks against an external dependency.",
    ["dependency"],
)


def record_http(method: str, route: str, status: int, duration_s: float) -> None:
    HTTP_REQUESTS.labels(method, route, str(status)).inc()
    HTTP_LATENCY.labels(method, route).observe(duration_s)


def record_stage(stage: str, outcome: StageOutcome, duration_s: float) -> None:
    STAGE_LATENCY.labels(stage, outcome).observe(duration_s)


def record_llm_call(
    model_alias: str,
    *,
    outcome: StageOutcome,
    duration_s: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
) -> None:
    LLM_REQUESTS.labels(model_alias, outcome).inc()
    LLM_LATENCY.labels(model_alias).observe(duration_s)
    if prompt_tokens:
        LLM_TOKENS.labels(model_alias, "prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(model_alias, "completion").inc(completion_tokens)
    if cost_usd:
        LLM_COST.labels(model_alias).inc(cost_usd)


def record_job(outcome: JobOutcome, *, attempt: int, duration_s: float) -> None:
    INGEST_JOBS.labels(outcome).inc()
    INGEST_JOB_LATENCY.labels(outcome).observe(duration_s)
    INGEST_JOB_ATTEMPT.labels(outcome).observe(attempt)


def record_dependency_error(dependency: Dependency) -> None:
    DEPENDENCY_ERRORS.labels(dependency).inc()
