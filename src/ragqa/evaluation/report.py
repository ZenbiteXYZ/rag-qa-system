"""Computed metrics of a run, and the Markdown reports built from them.

`build_metrics` is a function of the run directory's files only. The JSON it returns is what
`compare` reads, so two runs are compared without their raw records.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ragqa.evaluation.compare import paired_comparison
from ragqa.evaluation.golden import CATEGORIES
from ragqa.evaluation.metrics import (
    K_PRIMARY,
    THRESHOLD,
    aggregate,
    compute_run_metrics,
    percentile,
)
from ragqa.evaluation.records import METRIC_CODE_VERSION, RunData

PRIMARY = {
    "Retrieval": "coverage@3",
    "Filter extraction": "filter_exact_match",
    "End to end": "answer_correctness",
}
RETRIEVAL = (
    "coverage@3",
    "coverage@5",
    "mrr",
    "recall@3@0.8",
    "recall@3@0.5",
    "recall@3@1.0",
    "recall@5@0.8",
    "precision@3",
    "containment@3",
    "overlap@3",
    "context_coverage",
)
FILTERS = ("filter_precision", "filter_recall", "filter_f1", "filter_exact_match")
END_TO_END = (
    "answer_correctness",
    "faithfulness",
    "answer_relevancy",
    "citation_validity",
    "citation_support",
    "abstention_recall",
    "false_abstention",
    "version_accuracy",
)


def _series(run_metrics: Any, name: str) -> list[float]:
    return [m[name] for m in run_metrics.per_repetition.values() if m.get(name) is not None]


def build_metrics(run: RunData) -> dict[str, Any]:
    result = compute_run_metrics(run)
    questions = run.golden.by_id()
    by_category = {
        category: aggregate(
            result.per_question,
            [i for i in result.per_question if questions[i].question.category == category],
        )
        for category in CATEGORIES
        if any(questions[i].question.category == category for i in result.per_question)
    }
    stage_names = sorted(
        {
            n.removeprefix("latency_ms:")
            for m in result.per_repetition.values()
            for n in m
            if n.startswith("latency_ms:")
        }
    )

    def p50_p95(values: list[float]) -> dict[str, float | None]:
        return {"p50": percentile(values, 50), "p95": percentile(values, 95), "n": len(values)}

    outcomes = Counter(s["outcome"] for s in result.substitutions)
    replaced = [s for s in result.substitutions if s["correct"] is not None]
    return {
        "protocol_version": run.meta.protocol_version,
        "metric_code_version": METRIC_CODE_VERSION,
        "run_id": run.meta.run_id,
        "recomputed_from": run.meta.recomputed_from,
        "golden_hash": run.meta.golden_hash,
        "config_name": run.meta.name,
        "split": run.meta.split,
        "repetitions": run.meta.repetitions,
        "questions": len(result.per_question),
        "requests": len(run.records),
        "errors": result.errors,
        "degraded_requests": int(sum(_series(result, "degraded"))),
        "overall": aggregate(result.per_question),
        "by_category": by_category,
        "noise": result.noise,
        "latency": {
            "client_ms": p50_p95(_series(result, "latency_client_ms")),
            "api_ms": p50_p95(_series(result, "latency_api_ms")),
            "stages": {s: p50_p95(_series(result, f"latency_ms:{s}")) for s in stage_names},
        },
        "substitutions": {
            "outcomes": dict(outcomes),
            "replacements_correct": sum(bool(s["correct"]) for s in replaced),
            "replacements": len(replaced),
        },
        "per_question": result.per_question,
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """A Markdown table as lines, followed by a blank line."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return [*lines, ""]


def _mean(overall: Mapping[str, Any], name: str) -> float | None:
    value = (overall.get(name) or {}).get("mean")
    return value if isinstance(value, float) else None


def _metric_table(title: str, names: Sequence[str], metrics: Mapping[str, Any]) -> list[str]:
    overall, noise = metrics["overall"], metrics["noise"]
    rows = []
    for name in names:
        a = overall.get(name)
        if a is None:
            continue
        rows.append([name, fmt(a["mean"]), a["n"], fmt(a["stdev"]), fmt(noise.get(name))])
    if not rows:
        return []
    header = ["Metric", "Mean over questions", "n", "Stdev", "Run-to-run noise"]
    return [f"### {title}", "", *table(header, rows)]


def _tokens_per_request(overall: Mapping[str, Any]) -> float:
    return (_mean(overall, "tokens_prompt") or 0.0) + (_mean(overall, "tokens_completion") or 0.0)


def _header(metrics: Mapping[str, Any]) -> list[str]:
    return [
        f"# Run {metrics['run_id']}",
        "",
        f"Protocol v{metrics['protocol_version']}, metric code v{metrics['metric_code_version']}, "
        f"golden set `{metrics['golden_hash'][:12]}`, config `{metrics['config_name']}`, "
        f"split `{metrics['split']}`, {metrics['repetitions']} repetitions.",
        f"{metrics['questions']} questions, {metrics['requests']} requests, "
        f"{len(metrics['errors'])} failed, {metrics['degraded_requests']} with a degraded stage.",
        "",
        "Means are taken per question first (over the repetitions), then over questions; "
        "`n` is the number of questions the metric is defined for. Failed requests have no "
        f"metrics and are listed below. Retrieval metrics use k={K_PRIMARY} and threshold "
        f"{THRESHOLD} (ADR-0019).",
        "",
    ]


def _primary(metrics: Mapping[str, Any]) -> list[str]:
    overall = metrics["overall"]
    rows: list[Sequence[object]] = [
        [level, name, fmt(_mean(overall, name))] for level, name in PRIMARY.items()
    ]
    rows.append(["System", "p95 latency (ms)", fmt(metrics["latency"]["client_ms"]["p95"], 0)])
    rows.append(["System", "tokens per request", fmt(_tokens_per_request(overall), 0)])
    return ["## Primary metrics", "", *table(["Level", "Metric", "Value"], rows)]


def _substitutions(metrics: Mapping[str, Any]) -> list[str]:
    sub = metrics["substitutions"]
    if not sub["outcomes"]:
        return []
    outcomes = ", ".join(f"{k} {v}" for k, v in sorted(sub["outcomes"].items()))
    return [
        f"Extracted values: {outcomes}. Replacements that were the expected value: "
        f"{sub['replacements_correct']} of {sub['replacements']}.",
        "",
    ]


def _system(metrics: Mapping[str, Any]) -> list[str]:
    lat, overall = metrics["latency"], metrics["overall"]

    def row(label: str, v: Mapping[str, Any]) -> list[object]:
        return [label, fmt(v["p50"], 0), fmt(v["p95"], 0), v["n"]]

    rows = [
        row("End-to-end latency, client (ms)", lat["client_ms"]),
        row("Pipeline time, sum of stages (ms)", lat["api_ms"]),
    ]
    rows += [row(f"Stage {stage} (ms)", v) for stage, v in lat["stages"].items()]
    per_request = [
        [label, fmt(_mean(overall, name), 1)]
        for label, name in (
            ("Prompt tokens", "tokens_prompt"),
            ("Completion tokens", "tokens_completion"),
            ("LLM calls", "llm_calls"),
        )
    ]
    return [
        "## System",
        "",
        *table(["Quantity", "p50", "p95", "n"], rows),
        *table(["Per request", "Mean"], per_request),
    ]


def _by_category(metrics: Mapping[str, Any]) -> list[str]:
    rows = []
    for category, agg in metrics["by_category"].items():
        cells = []
        for name in PRIMARY.values():
            a = agg.get(name)
            cells.append(f"{fmt(a['mean'])} (n={a['n']})" if a and a["mean"] is not None else "-")
        rows.append([category, *cells])
    return [
        "## By category (descriptive only: 3 questions per category and split)",
        "",
        *table(["Category", *PRIMARY.values()], rows),
    ]


def render_report(metrics: Mapping[str, Any]) -> str:
    lines = [
        *_header(metrics),
        *_primary(metrics),
        "## Retrieval",
        "",
        *_metric_table("Retrieval metrics", RETRIEVAL, metrics),
        "## Filter extraction",
        "",
        *_metric_table("Filter metrics", FILTERS, metrics),
        *_substitutions(metrics),
        "## End to end",
        "",
        *_metric_table("Answer metrics", END_TO_END, metrics),
        *_system(metrics),
        *_by_category(metrics),
    ]
    if metrics["errors"]:
        lines += ["## Failed requests", ""]
        for e in metrics["errors"][:20]:
            lines.append(f"- {e['question_id']} (repetition {e['repetition']}): {e['error'][:200]}")
    return "\n".join(lines) + "\n"


COMPARED = (*PRIMARY.values(), "coverage@5", "mrr", "answer_relevancy", "faithfulness")


def render_comparison(
    base: Mapping[str, Any],
    other: Mapping[str, Any],
    names: Sequence[str] = COMPARED,
    *,
    seed: int = 0,
) -> str:
    """Paired comparison of two runs on the same questions. Raises ValueError when the runs
    are not comparable (ML_DESIGN section 6, rule 4)."""
    for key in ("protocol_version", "golden_hash"):
        if base[key] != other[key]:
            raise ValueError(f"runs differ in {key}: recompute from the stored records or re-run")
    rows = []
    for name in names:
        noise_values = [
            v for v in (base["noise"].get(name), other["noise"].get(name)) if v is not None
        ]
        result = paired_comparison(
            base["per_question"],
            other["per_question"],
            name,
            noise=max(noise_values) if noise_values else None,
            seed=seed,
        )
        if result is None:
            continue
        rows.append(
            [
                name,
                fmt(result.mean_base),
                fmt(result.mean_other),
                f"{result.mean_diff:+.3f}",
                f"[{result.ci_low:+.3f}, {result.ci_high:+.3f}]",
                f"{result.better} / {result.worse} / {result.unchanged}",
                result.n,
                fmt(result.noise),
                "yes" if result.distinguishable else "no",
            ]
        )
    header = [
        "Metric",
        "Baseline",
        "Other",
        "Difference",
        "95% interval",
        "Better / worse / same",
        "n",
        "Noise",
        "Distinguishable",
    ]
    return "\n".join(
        [
            f"# {other['run_id']} against baseline {base['run_id']}",
            "",
            f"Protocol v{base['protocol_version']}, golden set `{base['golden_hash'][:12]}`. "
            "Questions are compared one by one; the interval is a bootstrap over questions "
            "(95%). A difference is marked distinguishable only if the interval excludes 0 and "
            "it is larger than the run-to-run noise (mean absolute difference between the "
            "repetitions). The baseline is always shown.",
            "",
            *table(header, rows),
            f"Failed requests: baseline {len(base['errors'])}, other {len(other['errors'])}. "
            "Questions with a missing value in one run are left out of that metric.",
            "",
        ]
    )


def dump(metrics: Mapping[str, Any]) -> str:
    return json.dumps(metrics, indent=1, sort_keys=True)
