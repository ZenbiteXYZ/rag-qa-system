"""The evaluation chain on small fake data: golden set, run, judge, metrics, report, compare."""

import json
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragqa.domain.pipeline import PipelineConfig, PipelineState, StageTiming
from ragqa.domain.retrieval import RetrievedChunk
from ragqa.evaluation.golden import PROTOCOL_VERSION, load_golden
from ragqa.evaluation.judge import judge_jobs, run_jobs
from ragqa.evaluation.manifest import IngestManifest, ManifestVersion
from ragqa.evaluation.metrics import state_of
from ragqa.evaluation.records import RunDir, append_jsonl, load_run
from ragqa.evaluation.report import build_metrics, dump, render_comparison, render_report
from ragqa.evaluation.runner import (
    check_against_manifest,
    config_hash,
    create_run,
    load_eval_config,
    request_version,
)
from tests.query_fakes import INDEX, FakeLLM

REL_OLD, REL_NEW = "release-1.32", "release-1.34"
DOC = "tasks/drain.md"
TEXTS = {
    (DOC, REL_OLD): "# Drain\n\nUse kubectl drain to empty a node.\n\nThe flag is --force.\n",
    (
        DOC,
        REL_NEW,
    ): "# Drain\n\nUse kubectl drain to empty a node.\n\nThe flag is --disable-eviction.\n",
}

GOLDEN = """
- id: q1
  category: single_section
  split: dev
  question: How do I empty a node?
  reference_answer: Use kubectl drain.
  evidence:
    - {document: tasks/drain.md, version: release-1.34, quote: "Use kubectl drain to empty a node."}
- id: q2
  category: version_sensitive
  split: dev
  question: In 1.32, which flag skips eviction?
  release: release-1.32
  reference_answer: --force.
  evidence:
    - {document: tasks/drain.md, version: release-1.32, quote: "The flag is --force."}
- id: q3
  category: unanswerable
  split: test
  question: How do I train a model?
  reference_answer: The documents do not say.
"""

DOC_ID = uuid.UUID(int=10)
V_OLD, V_NEW = uuid.UUID(int=11), uuid.UUID(int=12)
MANIFEST = IngestManifest(
    index="base",
    corpus_releases=(REL_OLD, REL_NEW),
    versions=(
        ManifestVersion(
            document=DOC, release=REL_OLD, document_id=DOC_ID, version_id=V_OLD, version=1
        ),
        ManifestVersion(
            document=DOC, release=REL_NEW, document_id=DOC_ID, version_id=V_NEW, version=2
        ),
    ),
)


def provider(document: str, version: str) -> str:
    return TEXTS[(document, version)]


@pytest.fixture
def golden_file(tmp_path: Path) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(GOLDEN, encoding="utf-8")
    return path


def chunk_at(
    text_key: tuple[str, str], needle: str, version_id: uuid.UUID, n: int = 0
) -> RetrievedChunk:
    text = TEXTS[text_key]
    start = text.index(needle)
    return RetrievedChunk(
        point_id=f"p{n}",
        document_id=DOC_ID,
        version_id=version_id,
        version=2 if version_id == V_NEW else 1,
        title="Drain",
        section_path=("Drain",),
        chunk_index=n,
        start=start,
        end=start + len(needle),
        text=needle,
        has_code=False,
        token_count=9,
        score=1.0,
    )


def api_response(state: PipelineState) -> dict[str, Any]:
    return {"status": "answer", "debug": state.model_dump(mode="json")}


def state_for(question: str, version_ok: bool = True) -> PipelineState:
    config = PipelineConfig(default_model="haiku-5.5")
    base = {"question": question, "index": INDEX, "config": config}
    timings = (StageTiming(stage="retrieve", ms=30.0), StageTiming(stage="generate", ms=700.0))
    if "empty a node" in question:
        chunk = chunk_at((DOC, REL_NEW), "Use kubectl drain to empty a node.", V_NEW)
        return PipelineState(
            **base,
            candidates=(chunk,),
            context_chunks=(chunk,),
            answer="Use kubectl drain [1].",
            citation_numbers=(1,),
            timings=timings,
        )
    if "which flag" in question:
        release = REL_OLD if version_ok else REL_NEW
        vid = V_OLD if version_ok else V_NEW
        needle = "The flag is --force." if version_ok else "The flag is --disable-eviction."
        chunk = chunk_at((DOC, release), needle, vid)
        return PipelineState(
            **base,
            candidates=(chunk,),
            context_chunks=(chunk,),
            answer="--force [1]",
            citation_numbers=(1,),
            timings=timings,
        )
    return PipelineState(
        **base,
        answer="I cannot answer this from the available documents.",
        abstained=True,
        timings=timings,
    )


def fake_api(
    seen: list[dict[str, Any]], *, fail_for: str | None = None, version_ok: bool = True
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if fail_for and fail_for in body["question"]:
            return httpx.Response(503, json={"error": {"code": "llm_unavailable"}})
        return httpx.Response(200, json=api_response(state_for(body["question"], version_ok)))

    return httpx.MockTransport(handler)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.yaml"
    path.write_text(
        "name: baseline\nindex: base\npipeline: {retrieve: {mode: dense}}\n"
        "pins: {embedder: {revision: abc}}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    MANIFEST.save(path)
    return path


async def make_run(
    tmp_path: Path,
    golden_file: Path,
    config_file: Path,
    manifest_file: Path,
    name: str = "reports",
    **api: Any,
) -> tuple[Path, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    golden = load_golden(golden_file)
    path = await create_run(
        api_url="http://api",
        golden=golden,
        snapshot=golden.snapshot(provider),
        config_path=config_file,
        manifest_path=manifest_file,
        out=tmp_path / name,
        split="all",
        repetitions=2,
        transport=fake_api(seen, **api),
    )
    return path, seen


# --- golden set ---


def test_golden_loading_validation_and_hash(golden_file: Path, tmp_path: Path) -> None:
    golden = load_golden(golden_file)

    assert [q.id for q in golden.questions] == ["q1", "q2", "q3"]
    assert [q.id for q in golden.select("dev")] == ["q1", "q2"]
    assert golden.select("all") == golden.questions
    reordered = tmp_path / "other.yaml"
    reordered.write_text(GOLDEN.replace("{document", "{ document"), encoding="utf-8")
    assert load_golden(reordered).hash == golden.hash  # formatting does not change the hash
    assert any("expected 3" in issue for issue in golden.check_composition())


def test_golden_shape_errors_name_the_question(tmp_path: Path) -> None:
    common = "split: dev, question: q, reference_answer: r"
    evidence = "evidence: [{document: d, version: v, quote: x}]"
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"- {{id: a, category: follow_up, {common}, {evidence}}}\n"
        f"- {{id: b, category: unanswerable, {common}, {evidence}}}\n"
        f"- {{id: c, category: single_section, {common}}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_golden(bad)

    text = str(error.value)
    assert "a:" in text and "needs a history" in text
    assert "b:" in text and "no evidence" in text
    assert "c:" in text and "at least one required evidence" in text


def test_snapshot_resolves_spans_and_rejects_bad_quotes(golden_file: Path, tmp_path: Path) -> None:
    golden = load_golden(golden_file)

    snapshot = golden.snapshot(provider)

    q1 = snapshot.by_id()["q1"]
    span = q1.spans[0]
    assert span is not None
    assert TEXTS[(DOC, REL_NEW)][span.start : span.end] == "Use kubectl drain to empty a node."
    assert snapshot.protocol_version == PROTOCOL_VERSION and snapshot.golden_hash == golden.hash
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        GOLDEN.replace("The flag is --force.", "The flag is --nothing."), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="NOT_FOUND"):
        load_golden(broken).snapshot(provider)


# --- manifest and request building ---


def test_release_maps_to_a_version_number() -> None:
    golden = load_golden_text()
    q2 = golden["q2"]
    assert request_version(q2, MANIFEST) == (1, None)
    assert request_version(golden["q1"], MANIFEST) == (None, None)  # no release: the default
    assert check_against_manifest(list(golden.values()), MANIFEST) == []
    empty = IngestManifest(index="base", corpus_releases=(), versions=())
    assert "not in the ingest manifest" in check_against_manifest([q2], empty)[0]


def load_golden_text() -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "g.yaml"
        path.write_text(GOLDEN, encoding="utf-8")
        return {q.id: q for q in load_golden(path).questions}


# --- the run ---


async def test_run_sends_stateless_requests_and_stores_everything(
    tmp_path: Path, golden_file: Path, config_file: Path, manifest_file: Path
) -> None:
    path, seen = await make_run(tmp_path, golden_file, config_file, manifest_file)

    run = load_run(path)
    assert len(run.records) == 6  # 3 questions x 2 repetitions
    assert {r.repetition for r in run.records} == {0, 1}
    by_question = {b["question"]: b for b in seen}
    assert by_question["In 1.32, which flag skips eviction?"]["version"] == 1
    assert by_question["How do I empty a node?"]["version"] is None
    assert all(b["debug"] is True and b["history"] == [] and b["index"] == "base" for b in seen)
    assert seen[0]["config"] == {"retrieve": {"mode": "dense"}}
    assert run.meta.protocol_version == PROTOCOL_VERSION and run.meta.repetitions == 2
    assert run.meta.models == {"embedder": {"revision": "abc"}}
    assert run.meta.config_hash == config_hash(load_eval_config(config_file))
    assert run.golden.questions[0].spans[0] is not None
    assert (path / "finished.json").exists() and (path / "config.yaml").exists()
    assert json.loads((path / "finished.json").read_text())["errors"] == 0


async def test_failed_requests_are_stored_not_dropped(
    tmp_path: Path, golden_file: Path, config_file: Path, manifest_file: Path
) -> None:
    path, _ = await make_run(
        tmp_path, golden_file, config_file, manifest_file, fail_for="train a model"
    )

    run = load_run(path)
    failed = [r for r in run.records if r.error]
    assert len(failed) == 2 and all(r.http_status == 503 and r.response is None for r in failed)
    metrics = build_metrics(run)
    assert len(metrics["errors"]) == 2 and metrics["requests"] == 6
    assert "q3" in metrics["per_question"]  # kept, with system metrics only
    assert "coverage@3" not in metrics["per_question"]["q3"]


async def test_a_run_directory_is_never_reused(
    tmp_path: Path, golden_file: Path, config_file: Path, manifest_file: Path
) -> None:
    path, _ = await make_run(tmp_path, golden_file, config_file, manifest_file)
    run = RunDir(path)

    with pytest.raises(FileExistsError):
        run.write_once(run.meta, "{}")


async def test_a_manifest_that_does_not_match_stops_before_any_request(
    tmp_path: Path, golden_file: Path, config_file: Path
) -> None:
    other = IngestManifest(index="base", corpus_releases=(), versions=())
    manifest = tmp_path / "m.json"
    other.save(manifest)
    seen: list[dict[str, Any]] = []
    golden = load_golden(golden_file)

    with pytest.raises(ValueError, match="not in the ingest manifest"):
        await create_run(
            api_url="http://api",
            golden=golden,
            snapshot=golden.snapshot(provider),
            config_path=config_file,
            manifest_path=manifest,
            out=tmp_path / "reports",
            split="all",
            repetitions=1,
            transport=fake_api(seen),
        )
    assert seen == []


# --- metrics, judge, report and comparison ---


async def judge_run(path: Path, llm: FakeLLM) -> None:
    run, run_dir = load_run(path), RunDir(path)
    questions = run.golden.by_id()
    jobs = []
    for record in run.records:
        state = state_of(record)
        if state is None or state.answer is None:
            continue
        q = questions[record.question_id].question
        for metric, messages, schema, score_of in judge_jobs(
            record, state, q.question, q.reference_answer
        ):
            jobs.append(
                ((record.question_id, record.repetition), metric, messages, schema, score_of)
            )
    for judged in await run_jobs(llm, jobs, model="gpt-6-luna", seed=7):
        append_jsonl(run_dir.judge, judged)


def judge_llm() -> FakeLLM:
    return FakeLLM(
        structured=[
            json.dumps(
                {
                    "verdict": "correct",
                    "reason": "ok",
                    "score": 5,
                    "claims": [{"claim": "c", "supported": True}],
                }
            )
        ]
    )


async def test_metrics_from_the_stored_run_only(
    tmp_path: Path, golden_file: Path, config_file: Path, manifest_file: Path
) -> None:
    path, _ = await make_run(tmp_path, golden_file, config_file, manifest_file)
    await judge_run(path, judge_llm())

    metrics = build_metrics(load_run(path))  # no corpus, no API: records and golden only

    overall = metrics["overall"]
    assert overall["coverage@3"]["mean"] == 1.0 and overall["coverage@3"]["n"] == 2  # q1, q2
    assert overall["abstention_recall"]["mean"] == 1.0 and overall["abstention_recall"]["n"] == 1
    assert overall["version_accuracy"]["mean"] == 1.0  # q2 cites 1.32 as asked
    assert overall["answer_correctness"]["n"] == 3 and overall["answer_correctness"]["mean"] == 1.0
    assert metrics["latency"]["stages"]["generate"]["p50"] == 700.0
    assert metrics["latency"]["client_ms"]["n"] == 6
    assert metrics["noise"]["coverage@3"] == 0.0  # the fake API answers the same twice
    assert set(metrics["by_category"]) == {"single_section", "version_sensitive", "unanswerable"}
    json.loads(dump(metrics))


async def test_report_and_paired_comparison(
    tmp_path: Path, golden_file: Path, config_file: Path, manifest_file: Path
) -> None:
    base_path, _ = await make_run(tmp_path, golden_file, config_file, manifest_file, "a")
    other_path, _ = await make_run(
        tmp_path, golden_file, config_file, manifest_file, "b", version_ok=False
    )
    base = build_metrics(load_run(base_path))
    other = build_metrics(load_run(other_path))

    report = render_report(base)
    assert "## Primary metrics" in report and "coverage@3" in report and "p95 latency" in report
    assert "descriptive only" in report

    assert other["overall"]["version_accuracy"]["mean"] == 0.0  # q2 now cites the wrong release
    text = render_comparison(base, other, ["coverage@3", "version_accuracy"])
    assert "against baseline" in text and "version_accuracy" in text

    other["golden_hash"] = "different"
    with pytest.raises(ValueError, match="golden_hash"):
        render_comparison(base, other)
