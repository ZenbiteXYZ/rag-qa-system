"""Checking extracted filter values against the values that exist (ADR-0008).

Pure functions, so the rules are testable without a database. The thresholds come from
`SelfQueryConfig` and are part of the pipeline config, so they are logged with every run.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from ragqa.domain.pipeline import SelfQueryConfig
from ragqa.domain.retrieval import KnownDocument

_WORD = re.compile(r"[^\W_]+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def similarity(value: str, candidate: str) -> float:
    """From 0 to 1. A value that is a whole-word part of the candidate ("Petrov" in
    "I. Petrov") scores at least 0.9, since users give surnames and short names."""
    a, b = normalize(value), normalize(candidate)
    if not a or not b:
        return 0.0
    score = SequenceMatcher(None, a, b).ratio()
    if a != b and f" {a} " in f" {b} ":
        score = max(score, 0.9)
    return score


@dataclass(frozen=True)
class Resolution:
    outcome: Literal["exact", "substituted", "asked", "unresolved"]
    resolved: str | None = None
    candidates: tuple[str, ...] = ()


def resolve_value(value: str, known: Sequence[str], config: SelfQueryConfig) -> Resolution:
    scored = sorted(((similarity(value, k), k) for k in known), reverse=True)
    if scored and scored[0][0] == 1.0 and normalize(value) == normalize(scored[0][1]):
        return Resolution("exact", scored[0][1])
    if scored:
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= config.match_threshold and best_score - runner_up >= config.margin:
            return Resolution("substituted", best)
    candidates = tuple(k for s, k in scored if s >= config.candidate_floor)[: config.max_candidates]
    if candidates:
        return Resolution("asked", candidates=candidates)
    return Resolution("unresolved")


@dataclass(frozen=True)
class DocumentResolution:
    outcome: Literal["exact", "substituted", "asked", "unresolved"]
    live: tuple[KnownDocument, ...] = ()
    deleted: KnownDocument | None = None  # the title names a deleted document
    candidates: tuple[str, ...] = ()
    matched_title: str | None = None


def resolve_document(
    title: str, documents: Sequence[KnownDocument], config: SelfQueryConfig
) -> DocumentResolution:
    by_title: dict[str, list[KnownDocument]] = {}
    for doc in documents:
        by_title.setdefault(doc.title, []).append(doc)
    resolution = resolve_value(title, sorted(by_title), config)
    if resolution.outcome in ("asked", "unresolved") or resolution.resolved is None:
        return DocumentResolution(resolution.outcome, candidates=resolution.candidates)
    matches = by_title[resolution.resolved]
    live = tuple(d for d in matches if d.deleted_at is None)
    if live:
        return DocumentResolution(resolution.outcome, live=live, matched_title=resolution.resolved)
    return DocumentResolution(
        resolution.outcome, deleted=matches[0], matched_title=resolution.resolved
    )
