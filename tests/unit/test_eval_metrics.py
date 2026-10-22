import uuid
from datetime import date
from typing import Any

import pytest

from ragqa.domain.pipeline import (
    Citation,
    LlmCallRecord,
    PipelineConfig,
    PipelineState,
    StageTiming,
    Substitution,
)
from ragqa.domain.retrieval import MetadataFilter, RetrievedChunk
from ragqa.evaluation.golden import (
    Evidence,
    GoldenQuestion,
    SnapshotQuestion,
    SpanOut,
)
from ragqa.evaluation.manifest import IngestManifest, ManifestVersion
from ragqa.evaluation.metrics import (
    Loc,
    aggregate,
    applied_filters,
    contained,
    coverage,
    filter_metrics,
    merge,
    mrr,
    percentile,
    precision,
    repetition_metrics,
    retrieval_metrics,
    substitution_outcomes,
    touched,
    version_accuracy,
)
from ragqa.evaluation.records import JudgeRecord, QuestionRecord
from tests.query_fakes import INDEX

DOC, REL = "tasks/drain.md", "release-1.34"


def snap(
    spans: list[tuple[int, int]],
    *,
    category: str = "single_section",
    docs: list[tuple[str, str]] | None = None,
    **extra: Any,
) -> SnapshotQuestion:
    docs = docs or [(DOC, REL)] * len(spans)
    question = GoldenQuestion(
        id="q1",
        category=category,  # type: ignore[arg-type]
        question="How do I drain a node?",
        evidence=tuple(
            Evidence(document=d, version=r, quote=f"quote {i}") for i, (d, r) in enumerate(docs)
        )
        if category != "unanswerable"
        else (),
        reference_answer="Use kubectl drain.",
        split="dev",
        **extra,
    )
    return SnapshotQuestion(
        question=question, spans=tuple(SpanOut(start=s, end=e) for s, e in spans)
    )


def loc(start: int, end: int, doc: str = DOC, rel: str = REL) -> Loc:
    return Loc(doc, rel, start, end)


def test_merge_intervals() -> None:
    assert merge([(5, 10), (0, 3), (3, 4), (8, 12)]) == [(0, 4), (5, 12)]


def test_coverage_is_the_share_of_evidence_characters_in_the_union() -> None:
    q = snap([(100, 200)])
    assert coverage(q, [loc(50, 150)]) == 0.5
    assert coverage(q, [loc(50, 150), loc(140, 260)]) == 1.0
    assert coverage(q, [loc(0, 100)]) == 0.0
    assert coverage(q, [loc(100, 200, rel="release-1.32")]) == 0.0  # another release
    assert coverage(q, [loc(100, 200, doc="other.md")]) == 0.0


def test_multi_evidence_coverage_averages_the_items() -> None:
    q = snap([(0, 100), (500, 600)], docs=[(DOC, REL), ("tasks/other.md", REL)])
    chunks = [loc(0, 100), loc(500, 550, doc="tasks/other.md")]
    assert coverage(q, chunks) == 0.75


def test_alternative_evidence_is_not_counted() -> None:
    question = GoldenQuestion(
        id="q1",
        category="single_section",
        question="q",
        evidence=(
            Evidence(document=DOC, version=REL, quote="a"),
            Evidence(document="x.md", version=REL, quote="b", required=False),
        ),
        reference_answer="r",
        split="dev",
    )
    q = SnapshotQuestion(
        question=question, spans=(SpanOut(start=0, end=10), SpanOut(start=0, end=10))
    )
    assert coverage(q, [loc(0, 10)]) == 1.0


def test_mrr_uses_the_rank_where_the_threshold_is_reached() -> None:
    q = snap([(0, 100)])
    assert mrr(q, [loc(0, 50), loc(50, 100)]) == 0.5  # 0.5 after one chunk, 1.0 after two
    assert mrr(q, [loc(0, 100)]) == 1.0
    assert mrr(q, [loc(0, 79), loc(500, 600)]) == 0.0  # 0.79 never reaches 0.8


def test_recall_thresholds_and_k() -> None:
    q = snap([(0, 100)])
    ranked = [loc(500, 600), loc(600, 700), loc(700, 800), loc(0, 100)]

    metrics = retrieval_metrics(q, ranked, [loc(0, 100)])

    assert metrics["coverage@3"] == 0.0 and metrics["coverage@5"] == 1.0
    assert metrics["recall@3@0.8"] == 0.0 and metrics["recall@5@0.8"] == 1.0
    assert metrics["context_coverage"] == 1.0
    half = retrieval_metrics(q, [loc(0, 50)], [])
    assert (half["recall@3@0.5"], half["recall@3@0.8"], half["recall@3@1.0"]) == (1.0, 0.0, 0.0)


def test_precision_counts_evidence_characters_over_chunk_characters() -> None:
    q = snap([(100, 200)])
    assert precision(q, [loc(90, 210)]) == pytest.approx(100 / 120)
    # Overlapping chunks are merged first, so shared characters count once.
    assert precision(q, [loc(90, 150), loc(140, 210)]) == pytest.approx(100 / 120)
    assert precision(q, [loc(300, 400)]) == 0.0
    assert precision(q, []) is None


def test_containment_needs_one_chunk_overlap_needs_any() -> None:
    q = snap([(100, 200)])
    assert contained(q, [loc(90, 210)]) == 1.0
    split = [loc(90, 150), loc(150, 210)]
    assert contained(q, split) == 0.0 and touched(q, split) == 1.0
    assert touched(q, [loc(0, 50)]) == 0.0


def test_filter_metrics() -> None:
    expected = {("author", "I. Petrov"), ("tags", "ops")}
    exact = filter_metrics({("author", "i petrov"), ("tags", "ops")}, expected)
    assert exact["filter_exact_match"] == 1.0 and exact["filter_f1"] == 1.0

    extra = filter_metrics(
        {("author", "i petrov"), ("tags", "ops"), ("doc_type", "task")}, expected
    )
    assert extra["filter_precision"] == pytest.approx(2 / 3) and extra["filter_recall"] == 1.0
    assert extra["filter_exact_match"] == 0.0

    missed = filter_metrics({("author", "i petrov")}, expected)
    assert missed["filter_precision"] == 1.0 and missed["filter_recall"] == 0.5
    assert missed["filter_f1"] == pytest.approx(2 / 3)

    assert set(filter_metrics(set(), set()).values()) == {1.0}  # nothing applied, nothing expected
    spurious = filter_metrics({("author", "x")}, set())
    assert spurious["filter_precision"] == 0.0 and spurious["filter_f1"] == 0.0
    assert filter_metrics(set(), expected)["filter_f1"] == 0.0


def pipeline_state(**update: Any) -> PipelineState:
    config = update.pop(
        "config",
        PipelineConfig.model_validate({"default_model": "m", "self_query": {"enabled": True}}),
    )
    return PipelineState(question="q", index=INDEX, config=config, **update)


def test_applied_filters_and_substitution_outcomes() -> None:
    state = pipeline_state(
        resolved_filters=MetadataFilter(
            author="I. Petrov", tags=("ops",), date_from=date(2026, 1, 1)
        ),
        resolved_document_title="Drain a Node",
        substitutions=(
            Substitution(
                field="author", value="Petrov", outcome="substituted", resolved="I. Petrov"
            ),
            Substitution(field="tags", value="ops", outcome="exact", resolved="ops"),
            Substitution(field="author", value="Zhang", outcome="unresolved"),
        ),
    )

    assert applied_filters(state) == {
        ("author", "i petrov"),
        ("tags", "ops"),
        ("date_from", "2026-01-01"),
        ("document", "drain a node"),
    }
    outcomes = substitution_outcomes(state, {("author", "I. Petrov")})
    assert [(o["outcome"], o["correct"]) for o in outcomes] == [
        ("substituted", True),
        ("exact", None),
        ("unresolved", None),
    ]
    wrong = substitution_outcomes(state, {("author", "A. Ivanova")})
    assert wrong[0]["correct"] is False


def test_version_accuracy_requires_the_expected_release_only() -> None:
    q = snap([(0, 100)], category="version_sensitive", release=REL)
    assert version_accuracy(q, [loc(0, 50)]) == 1.0
    assert version_accuracy(q, [loc(0, 50), loc(0, 50, rel="release-1.32")]) == 0.0
    assert version_accuracy(q, [loc(0, 50, rel="release-1.32")]) == 0.0
    assert version_accuracy(q, []) == 0.0
    assert version_accuracy(q, [loc(0, 50, doc="unrelated.md", rel="release-1.32")]) == 0.0


# --- A whole repetition ---

DOC_ID, VERSION_ID = uuid.uuid4(), uuid.uuid4()
MANIFEST = IngestManifest(
    index="base",
    corpus_releases=(REL,),
    versions=(
        ManifestVersion(
            document=DOC, release=REL, document_id=DOC_ID, version_id=VERSION_ID, version=1
        ),
    ),
)


def retrieved(n: int, start: int, end: int, version_id: uuid.UUID = VERSION_ID) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=f"p{n}",
        document_id=DOC_ID,
        version_id=version_id,
        version=1,
        title="Drain",
        section_path=(),
        chunk_index=n,
        start=start,
        end=end,
        text="t",
        has_code=False,
        score=1.0,
    )


def record(
    state: PipelineState | None, *, error: str | None = None, latency: float = 900.0
) -> QuestionRecord:
    return QuestionRecord(
        question_id="q1",
        repetition=0,
        split="dev",
        category="single_section",
        request={},
        http_status=200 if state else 500,
        error=error,
        response={"status": "answer", "debug": state.model_dump(mode="json")} if state else None,
        client_latency_ms=latency,
        started_at="2026-10-21T00:00:00Z",
    )


def test_a_full_repetition_combines_all_levels() -> None:
    q = snap([(100, 200)])
    chunks = (retrieved(0, 90, 210), retrieved(1, 300, 400, uuid.uuid4()))
    state = pipeline_state(
        candidates=chunks,
        context_chunks=chunks,
        answer="Use drain [1][9]",
        citation_numbers=(1, 9),
        citations=(
            Citation(
                document_id=DOC_ID,
                version=1,
                title="Drain",
                section_path=(),
                chunk_index=0,
                quote="t",
            ),
        ),
        timings=(StageTiming(stage="retrieve", ms=40.0), StageTiming(stage="generate", ms=800.0)),
        llm_calls=(
            LlmCallRecord(
                stage="generate",
                model="m",
                resolved_model="p/m",
                prompt_tokens=500,
                completion_tokens=60,
            ),
        ),
        resolved_filters=MetadataFilter(),
    )
    judge = [
        JudgeRecord(
            question_id="q1",
            repetition=0,
            metric="answer_correctness",
            judge_model="j",
            rubric_version="1",
            score=0.5,
        )
    ]

    m = repetition_metrics(q, record(state), judge, MANIFEST)

    assert m["coverage@3"] == 1.0 and m["recall@3@0.8"] == 1.0 and m["mrr"] == 1.0
    assert m["precision@3"] == pytest.approx(100 / (120 + 100))  # the second chunk is unknown
    assert m["citation_validity"] == 0.5  # [9] points at no excerpt
    assert m["citation_support"] == 1.0 and m["false_abstention"] == 0.0
    assert m["answer_correctness"] == 0.5
    assert m["filter_exact_match"] == 1.0  # Self-Query on, nothing expected, nothing applied
    assert m["latency_api_ms"] == 840.0 and m["latency_ms:generate"] == 800.0
    assert m["tokens_prompt"] == 500.0 and m["llm_calls"] == 1.0 and m["degraded"] == 0.0
    assert m["latency_client_ms"] == 900.0 and m["error"] == 0.0


def test_a_failed_request_has_only_system_metrics() -> None:
    m = repetition_metrics(snap([(0, 10)]), record(None, error="http 503"), [], MANIFEST)
    assert m == {"latency_client_ms": 900.0, "error": 1.0}


def test_unanswerable_questions_measure_abstention() -> None:
    q = snap([], category="unanswerable")
    refusing = pipeline_state(
        answer="I cannot answer this from the available documents.", abstained=True
    )
    answering = pipeline_state(answer="It is 42.")

    assert repetition_metrics(q, record(refusing), [], MANIFEST)["abstention_recall"] == 1.0
    assert repetition_metrics(q, record(answering), [], MANIFEST)["abstention_recall"] == 0.0
    assert "coverage@3" not in repetition_metrics(q, record(refusing), [], MANIFEST)


def test_a_refusal_on_an_answerable_question_scores_zero_support() -> None:
    q = snap([(0, 100)])
    state = pipeline_state(
        answer="I cannot answer this from the available documents.",
        abstained=True,
        context_chunks=(retrieved(0, 0, 100),),
        citation_numbers=(1,),
    )

    m = repetition_metrics(q, record(state), [], MANIFEST)

    assert m["false_abstention"] == 1.0 and m["citation_support"] == 0.0


def test_aggregate_is_a_mean_over_questions_with_counts() -> None:
    per_question = {"a": {"x": 1.0, "y": None}, "b": {"x": 0.0, "y": 0.5}, "c": {"x": 0.5}}

    result = aggregate(per_question)

    assert result["x"]["mean"] == 0.5 and result["x"]["n"] == 3
    assert result["y"]["mean"] == 0.5 and result["y"]["n"] == 1
    assert aggregate(per_question, ["a", "b"])["x"]["mean"] == 0.5


def test_percentile_interpolates() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 25.0
    assert percentile(values, 95) == pytest.approx(38.5)
    assert percentile([5.0], 95) == 5.0 and percentile([], 50) is None
