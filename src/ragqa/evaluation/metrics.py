"""Metrics of evaluation protocol v1 (ADR-0019, ML_DESIGN section 4).

Every function here is a function of the run records and the golden snapshot only, never of a
live call, so a changed definition can be recomputed offline. A change to any definition bumps
`METRIC_CODE_VERSION` (records.py) and, when it changes what a protocol metric means, the
protocol version.

Definitions that ML_DESIGN leaves open are fixed here and noted next to the function:
- Retrieval metrics read `ranked` when a reranker ran, else `candidates` ("the ranked
  candidates"), in rank order. `context_coverage` reads the final context.
- Chunks and evidence are matched by character overlap on the same (document, release).
- A failed request has no metrics. Aggregates report `n` (answered questions) and the number
  of errors; comparisons only use questions that succeeded in both runs.
"""

import math
import statistics
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ragqa.domain.pipeline import PipelineState
from ragqa.domain.retrieval import RetrievedChunk
from ragqa.evaluation.golden import SnapshotQuestion
from ragqa.evaluation.manifest import IngestManifest, ManifestVersion
from ragqa.evaluation.records import JudgeRecord, QuestionRecord, RunData
from ragqa.query.filters import normalize

K_PRIMARY = 3
K_SENSITIVITY = 5
THRESHOLD = 0.8
THRESHOLDS = (0.5, 0.8, 1.0)
_EPS = 1e-9

Interval = tuple[int, int]


# --- Intervals ---


def merge(intervals: Sequence[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap(a: Interval, merged: Sequence[Interval]) -> int:
    return sum(max(0, min(a[1], e) - max(a[0], s)) for s, e in merged)


@dataclass(frozen=True)
class Loc:
    """Where a retrieved chunk is in the corpus."""

    document: str
    release: str
    start: int
    end: int


UNKNOWN = "?"


def locate(chunk: RetrievedChunk, versions: Mapping[uuid.UUID, ManifestVersion]) -> Loc:
    """A chunk of a version that is not in the manifest still counts its characters (it can
    only lower precision), but matches no evidence."""
    v = versions.get(chunk.version_id)
    if v is None:
        return Loc(UNKNOWN, UNKNOWN, chunk.start, chunk.end)
    return Loc(v.document, v.release, chunk.start, chunk.end)


def _spans_for(locs: Sequence[Loc], document: str, release: str) -> list[Interval]:
    return merge([(c.start, c.end) for c in locs if (c.document, c.release) == (document, release)])


def _evidence_items(q: SnapshotQuestion) -> list[tuple[str, str, Interval]]:
    """(document, release, span) of each required evidence item."""
    out = []
    for item, span in zip(q.question.evidence, q.spans, strict=True):
        if item.required and span is not None:
            out.append((item.document, item.version, (span.start, span.end)))
    return out


# --- Retrieval ---


def coverage(q: SnapshotQuestion, locs: Sequence[Loc]) -> float:
    """Average over the required evidence items of the share of the item's characters that the
    union of the chunks covers (ML_DESIGN section 3)."""
    items = _evidence_items(q)
    if not items:
        return 0.0
    shares = []
    for document, release, span in items:
        covered = overlap(span, _spans_for(locs, document, release))
        shares.append(covered / (span[1] - span[0]))
    return sum(shares) / len(shares)


def mrr(q: SnapshotQuestion, locs: Sequence[Loc]) -> float:
    """1 / r for the smallest r such that the top r chunks reach the threshold, else 0."""
    for r in range(1, len(locs) + 1):
        if coverage(q, locs[:r]) >= THRESHOLD - _EPS:
            return 1 / r
    return 0.0


def precision(q: SnapshotQuestion, locs: Sequence[Loc]) -> float | None:
    """Share of the characters of the (merged) chunks that lie in the required evidence. None
    when there are no chunks."""
    items = _evidence_items(q)
    groups: dict[tuple[str, str], list[Interval]] = defaultdict(list)
    for loc in locs:
        groups[(loc.document, loc.release)].append((loc.start, loc.end))
    total = hit = 0
    for (document, release), spans in groups.items():
        chunks = merge(spans)
        total += sum(e - s for s, e in chunks)
        for doc, rel, span in items:
            if (doc, rel) == (document, release):
                hit += overlap(span, chunks)
    return hit / total if total else None


def contained(q: SnapshotQuestion, locs: Sequence[Loc]) -> float:
    """1 if a single chunk contains every required evidence item, else 0."""
    items = _evidence_items(q)
    for loc in locs:
        if all(
            (loc.document, loc.release) == (d, r) and loc.start <= s[0] and s[1] <= loc.end
            for d, r, s in items
        ):
            return 1.0
    return 0.0


def touched(q: SnapshotQuestion, locs: Sequence[Loc]) -> float:
    """1 if any chunk overlaps any required evidence item, else 0."""
    for document, release, span in _evidence_items(q):
        if overlap(span, _spans_for(locs, document, release)) > 0:
            return 1.0
    return 0.0


def retrieval_metrics(
    q: SnapshotQuestion, ranked: Sequence[Loc], context: Sequence[Loc]
) -> dict[str, float | None]:
    top3, top5 = ranked[:K_PRIMARY], ranked[:K_SENSITIVITY]
    cov3, cov5 = coverage(q, top3), coverage(q, top5)
    out: dict[str, float | None] = {
        "coverage@3": cov3,
        "coverage@5": cov5,
        "mrr": mrr(q, ranked),
        "precision@3": precision(q, top3),
        "containment@3": contained(q, top3),
        "overlap@3": touched(q, top3),
        "context_coverage": coverage(q, context),
    }
    for t in THRESHOLDS:
        out[f"recall@3@{t}"] = float(cov3 >= t - _EPS)
    out["recall@5@0.8"] = float(cov5 >= THRESHOLD - _EPS)
    return out


# --- Filter extraction ---


def applied_filters(state: PipelineState) -> set[tuple[str, str]]:
    """What Self-Query applied after the checks of ADR-0008, as normalized (field, value)."""
    out: set[tuple[str, str]] = set()
    f = state.resolved_filters
    if f is not None:
        for field in ("doc_type", "author", "language"):
            value = getattr(f, field)
            if value:
                out.add((field, normalize(value)))
        out |= {("tags", normalize(t)) for t in f.tags}
        for field in ("date_from", "date_to"):
            value = getattr(f, field)
            if value:
                out.add((field, value.isoformat()))
    if state.resolved_document_title:
        out.add(("document", normalize(state.resolved_document_title)))
    return out


def normalized_pairs(pairs: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {(f, v if f.startswith("date") else normalize(v)) for f, v in pairs}


def filter_metrics(
    applied: set[tuple[str, str]], expected: set[tuple[str, str]]
) -> dict[str, float]:
    """Precision, recall, F1 and exact match of one question. Nothing applied and nothing
    expected scores 1 on all of them."""
    expected = normalized_pairs(expected)
    hits = len(applied & expected)
    p = hits / len(applied) if applied else 1.0
    r = hits / len(expected) if expected else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "filter_precision": p,
        "filter_recall": r,
        "filter_f1": f1,
        "filter_exact_match": float(applied == expected),
    }


def substitution_outcomes(
    state: PipelineState, expected: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    """The outcome of every extracted value, and for a replaced value whether the replacement
    is the expected one (ADR-0008, ML_DESIGN 'Substitution outcome')."""
    want = normalized_pairs(expected)
    out = []
    for s in state.substitutions:
        correct = None
        if s.outcome == "substituted" and s.resolved is not None:
            correct = (s.field, normalize(s.resolved)) in want
        out.append({"field": s.field, "outcome": s.outcome, "correct": correct})
    return out


# --- End to end (deterministic part) ---


def cited_locs(state: PipelineState, versions: Mapping[uuid.UUID, ManifestVersion]) -> list[Loc]:
    context = state.context_chunks
    return [
        locate(context[n - 1], versions) for n in state.citation_numbers if 1 <= n <= len(context)
    ]


def version_accuracy(q: SnapshotQuestion, cited: Sequence[Loc]) -> float:
    """1 if the answer cites the expected release of the evidence documents and no other
    release of them, else 0 (an answer without such a citation scores 0)."""
    expected = {(d, r) for d, r, _ in _evidence_items(q)}
    documents = {d for d, _ in expected}
    seen = [(c.document, c.release) for c in cited if c.document in documents]
    return float(bool(seen) and all(pair in expected for pair in seen))


def end_to_end_metrics(
    q: SnapshotQuestion, state: PipelineState, versions: Mapping[uuid.UUID, ManifestVersion]
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "citation_validity": None,
        "citation_support": None,
        "abstention_recall": None,
        "false_abstention": None,
        "version_accuracy": None,
    }
    if state.answer is None:
        return out
    if state.citation_numbers:
        out["citation_validity"] = len(state.citations) / len(state.citation_numbers)
    if q.question.answerable:
        cited = cited_locs(state, versions)
        out["citation_support"] = 0.0 if state.abstained else coverage(q, cited)
        out["false_abstention"] = float(state.abstained)
        if q.question.category == "version_sensitive":
            out["version_accuracy"] = 0.0 if state.abstained else version_accuracy(q, cited)
    else:
        out["abstention_recall"] = float(state.abstained)
    return out


# --- System ---


def system_metrics(record: QuestionRecord, state: PipelineState | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "latency_client_ms": record.client_latency_ms,
        "error": float(record.error is not None),
    }
    if state is None:
        return out
    out["latency_api_ms"] = sum(t.ms for t in state.timings)
    for t in state.timings:
        out[f"latency_ms:{t.stage}"] = t.ms
    out["tokens_prompt"] = float(sum(c.prompt_tokens or 0 for c in state.llm_calls))
    out["tokens_completion"] = float(sum(c.completion_tokens or 0 for c in state.llm_calls))
    out["llm_calls"] = float(len(state.llm_calls))
    out["degraded"] = float(bool(state.degraded))
    return out


# --- Per question and repetition ---


def state_of(record: QuestionRecord) -> PipelineState | None:
    """The pipeline state from a record's debug output, None for a failed request."""
    if record.response is None or record.response.get("debug") is None:
        return None
    return PipelineState.model_validate(record.response["debug"])


def judged_metrics(judge: Sequence[JudgeRecord]) -> dict[str, float | None]:
    return {j.metric: j.score for j in judge}


def repetition_metrics(
    q: SnapshotQuestion,
    record: QuestionRecord,
    judge: Sequence[JudgeRecord],
    manifest: IngestManifest,
) -> dict[str, float | None]:
    """All metrics of one question in one repetition. None means not applicable."""
    state = state_of(record)
    out: dict[str, float | None] = system_metrics(record, state)
    if state is None:
        return out
    versions = manifest.by_version_id()
    if q.question.answerable:
        ranked_chunks = state.ranked or state.candidates
        out |= retrieval_metrics(
            q,
            [locate(c, versions) for c in ranked_chunks],
            [locate(c, versions) for c in state.context_chunks],
        )
    if state.config.self_query.enabled:
        expected = q.question.expected_filters.pairs()
        out |= filter_metrics(applied_filters(state), expected)
    out |= end_to_end_metrics(q, state, versions)
    out |= judged_metrics(judge)
    return out


# --- Aggregation ---


def mean_of(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear interpolation between the closest ranks."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass
class RunMetrics:
    per_repetition: dict[tuple[str, int], dict[str, float | None]]
    per_question: dict[str, dict[str, float | None]]  # mean over repetitions
    noise: dict[str, float]  # mean absolute difference between repetitions, per metric
    substitutions: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def compute_run_metrics(run: RunData) -> RunMetrics:
    questions = run.golden.by_id()
    judge_by_key: dict[tuple[str, int], list[JudgeRecord]] = defaultdict(list)
    for j in run.judge:
        judge_by_key[(j.question_id, j.repetition)].append(j)

    per_rep: dict[tuple[str, int], dict[str, float | None]] = {}
    substitutions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in run.records:
        q = questions[record.question_id]
        key = (record.question_id, record.repetition)
        per_rep[key] = repetition_metrics(q, record, judge_by_key.get(key, ()), run.manifest)
        if record.error is not None:
            errors.append(
                {
                    "question_id": record.question_id,
                    "repetition": record.repetition,
                    "error": record.error,
                }
            )
        state = state_of(record)
        if state is not None and state.config.self_query.enabled:
            for item in substitution_outcomes(state, q.question.expected_filters.pairs()):
                substitutions.append({"question_id": record.question_id, **item})

    by_question: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for (qid, _), metrics in sorted(per_rep.items()):
        by_question[qid].append(metrics)
    per_question: dict[str, dict[str, float | None]] = {}
    noise_samples: dict[str, list[float]] = defaultdict(list)
    for qid, reps in by_question.items():
        names = {m for r in reps for m in r}
        per_question[qid] = {m: mean_of([r.get(m) for r in reps]) for m in names}
        for m in names:
            values = [v for v in (r.get(m) for r in reps) if v is not None]
            if len(values) >= 2:
                noise_samples[m].append(abs(values[0] - values[1]))
    noise = {m: sum(v) / len(v) for m, v in noise_samples.items()}
    return RunMetrics(per_rep, per_question, noise, substitutions, errors)


def aggregate(
    per_question: Mapping[str, Mapping[str, float | None]], ids: Sequence[str] | None = None
) -> dict[str, dict[str, float | int | None]]:
    """Mean over questions per metric, with the number of questions it is defined for. The
    question is the unit (ML_DESIGN section 4)."""
    selected = [per_question[i] for i in (ids if ids is not None else per_question)]
    names = sorted({m for q in selected for m in q})
    out: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        values = [v for v in (q.get(name) for q in selected) if v is not None]
        out[name] = {
            "mean": sum(values) / len(values) if values else None,
            "n": len(values),
            "stdev": statistics.pstdev(values) if len(values) > 1 else None,
        }
    return out
