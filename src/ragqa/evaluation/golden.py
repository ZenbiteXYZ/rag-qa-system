"""The golden set: schema, loading and the checks of ADR-0019.

Questions are YAML lists in `eval/golden/` (ADR-0005). Evidence quotes are resolved against the
parser output by `ragqa.evaluation.evidence`; the resolved character spans are stored with the
questions in every run (`GoldenSet.snapshot`), so metrics never need the corpus again.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ragqa.evaluation.evidence import TextProvider, load_questions, validate_questions

PROTOCOL_VERSION = 1

Category = Literal[
    "single_section",
    "multi_section",
    "exact_token",
    "metadata_cue",
    "version_sensitive",
    "follow_up",
    "unanswerable",
]
CATEGORIES: tuple[str, ...] = (
    "single_section",
    "multi_section",
    "exact_token",
    "metadata_cue",
    "version_sensitive",
    "follow_up",
    "unanswerable",
)
PER_CATEGORY = {"dev": 3, "test": 3}  # ML_DESIGN section 3: 6 per category, split 3 and 3


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Evidence(_Model):
    document: str  # path below the release folder
    version: str  # release folder name, for example release-1.34
    quote: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    required: bool = True


class ExpectedFilters(_Model):
    """What Self-Query should extract. All empty means no filter is expected."""

    doc_type: str | None = None
    tags: tuple[str, ...] = ()
    author: str | None = None
    language: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    document: str | None = None  # a document the question names, by title

    def pairs(self) -> set[tuple[str, str]]:
        """(field, value) pairs, one per tag."""
        out: set[tuple[str, str]] = set()
        for field in ("doc_type", "author", "language", "document"):
            value = getattr(self, field)
            if value:
                out.add((field, str(value)))
        out |= {("tags", t) for t in self.tags}
        for field in ("date_from", "date_to"):
            value = getattr(self, field)
            if value:
                out.add((field, value.isoformat()))
        return out


class Turn(_Model):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class GoldenQuestion(_Model):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    category: Category
    question: str = Field(min_length=1)
    history: tuple[Turn, ...] = ()
    expected_filters: ExpectedFilters = ExpectedFilters()
    evidence: tuple[Evidence, ...] = ()
    reference_answer: str = Field(min_length=1)
    split: Literal["dev", "test"]
    # Release the question asks about, when it names one (version-sensitive questions). The
    # runner turns it into the `version` of the request using the ingest manifest.
    release: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> "GoldenQuestion":
        answerable = self.category != "unanswerable"
        required = [e for e in self.evidence if e.required]
        if answerable and not required:
            raise ValueError("an answerable question needs at least one required evidence item")
        if not answerable and self.evidence:
            raise ValueError("an unanswerable question has no evidence")
        if self.category == "follow_up" and not self.history:
            raise ValueError("a follow-up question needs a history")
        return self

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    @property
    def required_evidence(self) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if e.required)


class SpanOut(_Model):
    start: int
    end: int


class SnapshotQuestion(_Model):
    """A question with the resolved span of each evidence item (same order as `evidence`)."""

    question: GoldenQuestion
    spans: tuple[SpanOut | None, ...] = ()


class GoldenSnapshot(_Model):
    """Everything the metrics need from the golden set, stored in every run."""

    protocol_version: int
    golden_hash: str
    questions: tuple[SnapshotQuestion, ...]

    def by_id(self) -> dict[str, SnapshotQuestion]:
        return {q.question.id: q for q in self.questions}


class GoldenSet:
    def __init__(self, questions: list[GoldenQuestion], raw: list[Any]) -> None:
        self.questions = questions
        self._raw = raw

    @property
    def hash(self) -> str:
        """Content hash, independent of YAML formatting and question order."""
        payload = json.dumps(
            sorted((q.model_dump(mode="json") for q in self.questions), key=lambda d: d["id"]),
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def select(self, split: str | None) -> list[GoldenQuestion]:
        return [q for q in self.questions if split in (None, "all", q.split)]

    def check_composition(self) -> list[str]:
        """Issues with the shape of the set against ADR-0019: unique ids and 3 dev plus 3 test
        questions in each of the 7 categories."""
        issues: list[str] = []
        ids = Counter(q.id for q in self.questions)
        issues += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
        counts: Counter[tuple[str, str]] = Counter((q.category, q.split) for q in self.questions)
        for category in CATEGORIES:
            for split, expected in PER_CATEGORY.items():
                got = counts[(category, split)]
                if got != expected:
                    issues.append(f"{category}: {got} {split} questions, expected {expected}")
        return issues

    def snapshot(self, provider: TextProvider) -> GoldenSnapshot:
        """Resolve every quote against the corpus text and keep the spans. Raises ValueError
        listing every quote that does not resolve to exactly one place."""
        issues, spans = validate_questions(self._raw, provider)
        if issues:
            raise ValueError("\n".join(str(i) for i in issues))
        by_id = {q.id: q for q in self.questions}
        out = []
        for raw in self._raw:
            question = by_id[str(raw["id"])]
            resolved = tuple(
                SpanOut(start=s.start, end=s.end)
                if (s := spans.get((question.id, i))) is not None
                else None
                for i in range(len(question.evidence))
            )
            out.append(SnapshotQuestion(question=question, spans=resolved))
        return GoldenSnapshot(
            protocol_version=PROTOCOL_VERSION, golden_hash=self.hash, questions=tuple(out)
        )


def load_golden(path: Path) -> GoldenSet:
    """Read the YAML file or directory. Raises ValueError with every invalid question."""
    raw = load_questions(path)
    questions: list[GoldenQuestion] = []
    problems: list[str] = []
    for position, item in enumerate(raw):
        label = item.get("id", f"#{position}") if isinstance(item, Mapping) else f"#{position}"
        try:
            questions.append(GoldenQuestion.model_validate(item))
        except ValidationError as exc:
            for error in exc.errors():
                where = ".".join(str(p) for p in error["loc"])
                problems.append(f"{label}: {where}: {error['msg']}")
    if problems:
        raise ValueError("\n".join(problems))
    return GoldenSet(questions, raw)


def dump_yaml(questions: list[GoldenQuestion]) -> str:
    """For tests and tools that write a golden file."""
    return yaml.safe_dump(
        [q.model_dump(mode="json", exclude_defaults=True) for q in questions],
        sort_keys=False,
        allow_unicode=True,
    )
