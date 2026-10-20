import pytest
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import REGISTRY

from ragqa.observability.tracing import (
    inject_trace_headers,
    link_from_headers,
    llm_call,
    traced_stage,
)
from tests.conftest import LogLines


def _stage_count(stage: str, outcome: str) -> float:
    labels = {"stage": stage, "outcome": outcome}
    return REGISTRY.get_sample_value("ragqa_stage_duration_seconds_count", labels) or 0.0


def test_traced_stage_records_latency_and_binds_stage(log_lines: LogLines) -> None:
    before = _stage_count("t_rewrite", "ok")

    with traced_stage("t_rewrite", reads=["question"], writes=["rewritten_query"]) as span:
        span.set_inputs({"question": "q"})

    assert _stage_count("t_rewrite", "ok") == before + 1
    (line,) = [x for x in log_lines() if x["event"] == "stage_finished"]
    assert line["stage"] == "t_rewrite"
    assert line["outcome"] == "ok"


def test_traced_stage_records_error_and_reraises() -> None:
    before = _stage_count("t_failing", "error")

    with pytest.raises(ValueError), traced_stage("t_failing"):
        raise ValueError("boom")

    assert _stage_count("t_failing", "error") == before + 1


def test_llm_call_records_tokens(log_lines: LogLines) -> None:
    labels = {"model_alias": "t_generator", "kind": "completion"}
    before = REGISTRY.get_sample_value("ragqa_llm_tokens_total", labels) or 0.0

    with llm_call("t_generator") as call:
        call.prompt_tokens = 30
        call.completion_tokens = 7

    assert REGISTRY.get_sample_value("ragqa_llm_tokens_total", labels) == before + 7
    (line,) = [x for x in log_lines() if x["event"] == "llm_call"]
    assert line["prompt_tokens"] == 30


def test_trace_context_round_trip_through_headers() -> None:
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("publish") as span:
        headers = inject_trace_headers()
        ctx = span.get_span_context()

    # RabbitMQ may hand header values back as bytes.
    link = link_from_headers({k: v.encode() for k, v in headers.items()})

    assert link is not None
    assert link.trace_id == format(ctx.trace_id, "032x")
    assert link.span_id == format(ctx.span_id, "016x")


@pytest.mark.parametrize("headers", [None, {}, {"traceparent": "garbage"}])
def test_no_link_without_valid_traceparent(headers: dict[str, str] | None) -> None:
    assert link_from_headers(headers) is None
