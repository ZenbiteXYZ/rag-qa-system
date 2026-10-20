from prometheus_client import REGISTRY

from ragqa.observability import metrics

# Labels that are safe: bounded value sets. Anything else needs a deliberate review.
ALLOWED_LABELS = {
    "method",
    "route",
    "status",
    "stage",
    "outcome",
    "model_alias",
    "kind",
    "dependency",
}


def _value(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metric_labels_are_low_cardinality() -> None:
    for collector in vars(metrics).values():
        labelnames = getattr(collector, "_labelnames", None)
        if labelnames is not None:
            assert set(labelnames) <= ALLOWED_LABELS, collector


def test_record_http_counts_by_route_template() -> None:
    labels = {"method": "GET", "route": "/documents/{id}", "status": "200"}
    before = _value("ragqa_http_requests_total", labels)

    metrics.record_http("GET", "/documents/{id}", 200, 0.01)

    assert _value("ragqa_http_requests_total", labels) == before + 1


def test_record_llm_call_counts_tokens_and_cost() -> None:
    prompt = {"model_alias": "test-alias", "kind": "prompt"}
    before = _value("ragqa_llm_tokens_total", prompt)

    metrics.record_llm_call(
        "test-alias",
        outcome="ok",
        duration_s=0.5,
        prompt_tokens=100,
        completion_tokens=20,
        cost_usd=0.002,
    )

    assert _value("ragqa_llm_tokens_total", prompt) == before + 100
    assert _value("ragqa_llm_cost_usd_total", {"model_alias": "test-alias"}) >= 0.002
