"""The HTTP evaluation runner: sends every selected question to `POST /query` with the
pipeline config under test, and stores everything the metrics need (ML_DESIGN section 6).

The runner exercises the real service. It needs the API to run with
`RAGQA_ALLOW_QUERY_DEBUG=true`, because the stage outputs it records come from the `debug`
field of the response. Requests are stateless (inline history), so questions never see each
other. Failed requests are stored with their error, never retried and never dropped.
"""

import asyncio
import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from ragqa.evaluation.golden import (
    PROTOCOL_VERSION,
    GoldenQuestion,
    GoldenSet,
    GoldenSnapshot,
)
from ragqa.evaluation.manifest import IngestManifest
from ragqa.evaluation.records import (
    METRIC_CODE_VERSION,
    QuestionRecord,
    RunDir,
    RunMeta,
    append_jsonl,
)
from ragqa.query.prompts import PROMPT_VERSION

REQUEST_TIMEOUT_S = 180.0


class EvalConfig(BaseModel):
    """One configuration under test (`eval/configs/*.yaml`)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = ""
    index: str  # the index to query, by name
    pipeline: dict[str, Any] = Field(default_factory=dict)  # overrides of the default config
    # Pins that only the operator knows: the embedder revision, image tags and so on. Stored in
    # the run's meta as they are.
    pins: dict[str, Any] = Field(default_factory=dict)


def load_eval_config(path: Path) -> EvalConfig:
    return EvalConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def config_hash(config: EvalConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def git_info(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, cwd=cwd, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return sha, (bool(status) if status is not None else None)


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def request_version(
    question: GoldenQuestion, manifest: IngestManifest
) -> tuple[int | None, str | None]:
    """The `version` to send for a question that names a release, and an error if the release
    cannot be mapped (it is missing from the manifest or its evidence disagrees)."""
    if question.release is None:
        return None, None
    numbers = {
        manifest.version_number(e.document, question.release) for e in question.required_evidence
    }
    if None in numbers or len(numbers) != 1:
        return None, (
            f"release {question.release} maps to versions {sorted(n for n in numbers if n)}"
            " for the evidence documents; it must be exactly one"
        )
    return numbers.pop(), None


def build_request(
    question: GoldenQuestion, config: EvalConfig, version: int | None
) -> dict[str, Any]:
    return {
        "question": question.question,
        "index": config.index,
        "history": [t.model_dump() for t in question.history],  # [] is stateless
        "version": version,
        "config": config.pipeline,
        "debug": True,
    }


def check_against_manifest(
    questions: Sequence[GoldenQuestion], manifest: IngestManifest
) -> list[str]:
    """Evidence that points at a (document, release) the manifest does not have."""
    issues = []
    for q in questions:
        for e in q.evidence:
            if manifest.version_number(e.document, e.version) is None:
                issues.append(f"{q.id}: {e.document} at {e.version} is not in the ingest manifest")
        _, error = request_version(q, manifest)
        if error:
            issues.append(f"{q.id}: {error}")
    return issues


async def ask(
    http: httpx.AsyncClient,
    question: GoldenQuestion,
    repetition: int,
    config: EvalConfig,
    manifest: IngestManifest,
) -> QuestionRecord:
    version, error = request_version(question, manifest)
    request = build_request(question, config, version)
    started = now()
    start = time.perf_counter()
    status: int | None = None
    response: dict[str, Any] | None = None
    if error is None:
        try:
            reply = await http.post("/query", json=request, timeout=REQUEST_TIMEOUT_S)
            status = reply.status_code
            if status == 200:
                response = reply.json()
            else:
                error = f"http {status}: {reply.text[:500]}"
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
    return QuestionRecord(
        question_id=question.id,
        repetition=repetition,
        split=question.split,
        category=question.category,
        request=request,
        http_status=status,
        error=error,
        response=response,
        client_latency_ms=round((time.perf_counter() - start) * 1000, 1),
        started_at=started,
    )


async def execute(
    http: httpx.AsyncClient,
    questions: Sequence[GoldenQuestion],
    config: EvalConfig,
    manifest: IngestManifest,
    repetitions: int,
    run: RunDir,
    *,
    concurrency: int = 2,
    progress: Callable[[QuestionRecord], None] | None = None,
) -> list[QuestionRecord]:
    """Repetitions run one after the other, the questions of a repetition concurrently."""
    gate = asyncio.Semaphore(concurrency)
    records: list[QuestionRecord] = []

    async def one(q: GoldenQuestion, repetition: int) -> QuestionRecord:
        async with gate:
            return await ask(http, q, repetition, config, manifest)

    for repetition in range(repetitions):
        done = await asyncio.gather(*(one(q, repetition) for q in questions))
        for record in done:
            append_jsonl(run.records, record)
            records.append(record)
            if progress:
                progress(record)
    return records


def run_id(config: EvalConfig, started: datetime) -> str:
    return f"{config.name}-{started:%Y%m%dT%H%M%S}-{config_hash(config)[:8]}"


async def create_run(
    *,
    api_url: str,
    golden: GoldenSet,
    snapshot: GoldenSnapshot,
    config_path: Path,
    manifest_path: Path,
    out: Path,
    split: str,
    repetitions: int,
    concurrency: int = 2,
    repo_root: Path | None = None,
    progress: Callable[[QuestionRecord], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Path:
    """Run the selected questions and write a new run directory. Raises ValueError for
    problems found before any request is sent."""
    config = load_eval_config(config_path)
    manifest = IngestManifest.load(manifest_path)
    questions = golden.select(split)
    if not questions:
        raise ValueError(f"no questions in split {split!r}")
    problems = check_against_manifest(questions, manifest)
    if config.index != manifest.index:
        problems.append(f"config index {config.index!r} is not the manifest's {manifest.index!r}")
    if problems:
        raise ValueError("\n".join(problems))

    started = datetime.now(UTC)
    run = RunDir(out / run_id(config, started))
    run.path.mkdir(parents=True, exist_ok=False)
    chosen = {q.id for q in questions}
    selected = GoldenSnapshot(
        protocol_version=snapshot.protocol_version,
        golden_hash=snapshot.golden_hash,
        questions=tuple(q for q in snapshot.questions if q.question.id in chosen),
    )
    sha, dirty = git_info(repo_root)
    root = repo_root or Path.cwd()
    meta = RunMeta(
        run_id=run.path.name,
        protocol_version=PROTOCOL_VERSION,
        metric_code_version=METRIC_CODE_VERSION,
        name=config.name,
        config=config.model_dump(mode="json"),
        config_hash=config_hash(config),
        pipeline_config=config.pipeline,
        split=split,
        repetitions=repetitions,
        golden_hash=snapshot.golden_hash,
        git_sha=sha,
        git_dirty=dirty,
        uv_lock_hash=file_hash(root / "uv.lock"),
        prompt_version=PROMPT_VERSION,
        models=dict(config.pins),
        index={"name": manifest.index},
        api_url=api_url,
        started_at=started.isoformat(),
    )
    run.write_once(run.meta, meta.model_dump_json(indent=2))
    run.write_once(run.golden, selected.model_dump_json(indent=2))
    run.write_once(run.manifest, manifest.model_dump_json(indent=2))
    run.write_once(run.config, config_path.read_text(encoding="utf-8"))

    async with httpx.AsyncClient(base_url=api_url, transport=transport) as http:
        records = await execute(
            http,
            questions,
            config,
            manifest,
            repetitions,
            run,
            concurrency=concurrency,
            progress=progress,
        )
    run.write_once(run.path / "finished.json", json.dumps(finished(records), indent=2))
    return run.path


def finished(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    """What is only known after the run: when it ended, the effective pipeline config that the
    API reported, and the provider models that answered."""
    effective: dict[str, Any] | None = None
    resolved: set[str] = set()
    rerankers: set[str] = set()
    for r in records:
        debug = (r.response or {}).get("debug") or {}
        if effective is None and debug.get("config"):
            effective = debug["config"]
        resolved |= {c["resolved_model"] for c in debug.get("llm_calls", [])}
        if debug.get("reranker"):
            rerankers.add(debug["reranker"])
    return {
        "ended_at": now(),
        "requests": len(records),
        "errors": sum(r.error is not None for r in records),
        "effective_pipeline_config": effective,
        "resolved_models": sorted(resolved),
        "rerankers": sorted(rerankers),
    }
