"""Run records: everything a run stores so its metrics can be recomputed offline
(ML_DESIGN section 6).

A run is a directory that is written once. Metrics are a function of these files only.

    <run_id>/
        meta.json         parameters and tags of the run
        golden.json       the golden questions with resolved evidence spans (`GoldenSnapshot`)
        manifest.json     the ingest manifest (registry ids of the corpus)
        config.yaml       the evaluation config file that was used
        records.jsonl     one `QuestionRecord` per (question, repetition)
        judge.jsonl       one `JudgeRecord` per judged metric
        metrics.*.json    computed metrics, named with the protocol and metric-code versions
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragqa.evaluation.golden import GoldenSnapshot
from ragqa.evaluation.manifest import IngestManifest

METRIC_CODE_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuestionRecord(_Model):
    question_id: str
    repetition: int = Field(ge=0)
    split: str
    category: str
    request: dict[str, Any]
    http_status: int | None = None
    error: str | None = None
    # The `/query` response, including `debug` (the whole pipeline state).
    response: dict[str, Any] | None = None
    client_latency_ms: float
    started_at: str


class JudgeRecord(_Model):
    question_id: str
    repetition: int
    metric: str  # "answer_correctness", "faithfulness" or "answer_relevancy"
    judge_model: str  # alias
    resolved_model: str | None = None
    rubric_version: str
    seed: int | None = None
    verdict: dict[str, Any] | None = None
    score: float | None = None  # 0 to 1, from the verdict
    raw_text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class RunMeta(_Model):
    run_id: str
    protocol_version: int
    metric_code_version: int
    name: str  # of the evaluation config
    config: dict[str, Any]  # the evaluation config file
    config_hash: str
    pipeline_config: dict[str, Any]  # the effective PipelineConfig sent with every request
    split: str
    repetitions: int
    golden_hash: str
    git_sha: str | None = None
    git_dirty: bool | None = None
    uv_lock_hash: str | None = None
    prompt_version: str | None = None
    models: dict[str, Any] = Field(default_factory=dict)
    index: dict[str, Any] = Field(default_factory=dict)
    api_url: str | None = None
    started_at: str
    ended_at: str | None = None
    recomputed_from: str | None = None


class RunDir:
    """The files of one run. Writing never overwrites an existing file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def meta(self) -> Path:
        return self.path / "meta.json"

    @property
    def golden(self) -> Path:
        return self.path / "golden.json"

    @property
    def manifest(self) -> Path:
        return self.path / "manifest.json"

    @property
    def config(self) -> Path:
        return self.path / "config.yaml"

    @property
    def records(self) -> Path:
        return self.path / "records.jsonl"

    @property
    def judge(self) -> Path:
        return self.path / "judge.jsonl"

    def metrics(self, protocol: int, code: int) -> Path:
        return self.path / f"metrics.p{protocol}.m{code}.json"

    def report(self, protocol: int, code: int) -> Path:
        return self.path / f"report.p{protocol}.m{code}.md"

    def write_once(self, path: Path, text: str) -> None:
        if path.exists():
            raise FileExistsError(f"{path} exists; runs are never overwritten")
        path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, record: BaseModel) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> Iterator[T]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield model.model_validate_json(line)


class RunData(BaseModel):
    """A run loaded from disk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    meta: RunMeta
    golden: GoldenSnapshot
    manifest: IngestManifest
    records: list[QuestionRecord]
    judge: list[JudgeRecord]


def load_run(path: Path) -> RunData:
    run = RunDir(path)
    return RunData(
        meta=RunMeta.model_validate(json.loads(run.meta.read_text(encoding="utf-8"))),
        golden=GoldenSnapshot.model_validate(json.loads(run.golden.read_text(encoding="utf-8"))),
        manifest=IngestManifest.load(run.manifest),
        records=list(read_jsonl(run.records, QuestionRecord)),
        judge=list(read_jsonl(run.judge, JudgeRecord)),
    )
