import json
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragqa.evaluation import authors as authors_module
from ragqa.evaluation.calibrate import (
    CalibrationItem,
    calibrate,
    calibration_ok,
    evaluate,
    load_items,
)
from ragqa.evaluation.cli import parser
from ragqa.evaluation.judge import RUBRIC_VERSION
from ragqa.evaluation.load_corpus import Loader, pages
from ragqa.evaluation.mlflow_log import MlflowRest, flatten, metric_key, run_params
from ragqa.evaluation.records import RunMeta
from tests.query_fakes import FakeLLM


def items(labels: list[str]) -> list[CalibrationItem]:
    return [
        CalibrationItem(id=f"c{i}", question="q", reference_answer="r", answer="a", label=label)  # type: ignore[arg-type]
        for i, label in enumerate(labels)
    ]


def test_the_gate_passes_with_agreement_and_few_wrong_judged_correct() -> None:
    labels = ["correct"] * 8 + ["wrong"] * 2
    verdicts = ["correct"] * 8 + ["wrong"] * 2

    result = evaluate(items(labels), [verdicts, verdicts])

    assert result["gate_passed"] and result["self_agreement"] == 1.0
    assert result["passes"][0] == {"agreement": 1.0, "wrong_judged_correct": 0.0}


def test_the_gate_fails_on_low_agreement_or_on_wrong_answers_judged_correct() -> None:
    labels = ["correct"] * 5 + ["wrong"] * 5
    lenient = ["correct"] * 10  # 50% agreement and every wrong answer judged correct
    result = evaluate(items(labels), [lenient, lenient])
    assert not result["gate_passed"]
    assert result["passes"][0]["wrong_judged_correct"] == 1.0

    # 90% agreement overall, but one of the two wrong answers is judged correct: 50% > 10%.
    labels2 = ["correct"] * 8 + ["wrong"] * 2
    verdicts = ["correct"] * 9 + ["wrong"]
    assert not evaluate(items(labels2), [verdicts, verdicts])["gate_passed"]


def test_the_gate_needs_wrong_labels_and_one_failed_pass_fails_it() -> None:
    no_wrong = evaluate(items(["correct"] * 10), [["correct"] * 10] * 2)
    assert not no_wrong["gate_passed"] and any(
        "no answer is labeled wrong" in n for n in no_wrong["notes"]
    )

    labels = ["correct"] * 8 + ["wrong"] * 2
    good, bad = ["correct"] * 8 + ["wrong"] * 2, ["correct"] * 7 + [None] * 3
    assert not evaluate(items(labels), [good, bad])[
        "gate_passed"
    ]  # failed calls count as disagreement


async def test_calibration_runs_two_passes_and_the_result_gates_the_judge(tmp_path: Path) -> None:
    llm = FakeLLM(structured=[json.dumps({"verdict": "correct", "reason": "x"})])
    data = items(["correct"] * 8 + ["wrong"] * 2)

    result = await calibrate(llm, data, model="gpt-6-luna", seed=7)

    assert len(llm.calls) == 20 and len(result["verdicts"]) == 10  # 2 passes x 10 items
    assert result["judge_model"] == "gpt-6-luna" and result["rubric_version"] == RUBRIC_VERSION
    assert not result["gate_passed"]  # the fake judge calls everything correct

    path = tmp_path / "calibration.json"
    assert calibration_ok(path, model="gpt-6-luna")[0] is False  # no file
    path.write_text(json.dumps({**result, "gate_passed": True}), encoding="utf-8")
    assert calibration_ok(path, model="gpt-6-luna") == (True, "calibrated")
    assert calibration_ok(path, model="other-judge")[0] is False
    path.write_text(
        json.dumps({**result, "gate_passed": True, "rubric_version": "0"}), encoding="utf-8"
    )
    assert calibration_ok(path, model="gpt-6-luna")[0] is False
    path.write_text(json.dumps(result), encoding="utf-8")
    assert "did not pass" in calibration_ok(path, model="gpt-6-luna")[1]


def test_calibration_items_load_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "- {id: a, question: q, reference_answer: r, answer: x, label: partly_correct}\n"
    )
    assert load_items(path)[0].label == "partly_correct"


# --- MLflow over REST ---


def meta() -> RunMeta:
    return RunMeta(
        run_id="base-1",
        protocol_version=1,
        metric_code_version=1,
        name="base",
        config={},
        config_hash="h",
        pipeline_config={"retrieve": {"mode": "hybrid", "top_k": 10}},
        split="dev",
        repetitions=2,
        golden_hash="g" * 64,
        git_sha="abc",
        models={"embedder": {"revision": "r" * 600}},
        started_at="t",
    )


def test_metric_keys_and_params() -> None:
    assert metric_key("coverage@3") == "coverage_at_3"
    assert metric_key("recall@3@0.8") == "recall_at_3_at_0.8"
    assert metric_key("latency_ms:retrieve") == "latency_ms:retrieve"
    assert flatten("p", {"a": {"b": 1}, "c": "x"}) == {"p.a.b": "1", "p.c": "x"}
    params = run_params(meta())
    assert (
        params["pipeline.retrieve.mode"] == "hybrid" and params["pipeline.retrieve.top_k"] == "10"
    )
    assert len(params["pins.embedder.revision"]) == 500  # MLflow limits param values


def test_a_run_is_created_logged_uploaded_and_finished(tmp_path: Path) -> None:
    from ragqa.evaluation.mlflow_log import log_run

    (tmp_path / "meta.json").write_text("{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "records.jsonl").write_text("x")
    calls: list[tuple[str, str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content and request.method == "POST" else None
        calls.append((request.method, path, body))
        if path.endswith("experiments/get-by-name"):
            return httpx.Response(404, json={})
        if path.endswith("experiments/create"):
            return httpx.Response(200, json={"experiment_id": "7"})
        if path.endswith("runs/create"):
            info = {"run_id": "r1", "artifact_uri": "mlflow-artifacts:/7/r1/artifacts"}
            return httpx.Response(200, json={"run": {"info": info}})
        return httpx.Response(200, json={})

    run_id = log_run(
        tmp_path,
        tracking_uri="http://mlflow:5000",
        experiment="ragqa-eval",
        meta=meta(),
        aggregates={"coverage@3": 0.5},
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert run_id == "r1"
    puts = [p for m, p, _ in calls if m == "PUT"]
    assert puts == [
        "/api/2.0/mlflow-artifacts/artifacts/7/r1/artifacts/meta.json",
        "/api/2.0/mlflow-artifacts/artifacts/7/r1/artifacts/sub/records.jsonl",
    ]
    metrics = [b for _, p, b in calls if p.endswith("log-batch") and b and "metrics" in b]
    assert metrics[0]["metrics"][0]["key"] == "coverage_at_3"
    assert calls[-1][1].endswith("runs/update") and calls[-1][2]["status"] == "FINISHED"
    assert MlflowRest  # the class is the public surface used above


# --- authors ---


def test_author_mapping_is_deterministic_and_independent_of_input_order() -> None:
    paths = ["b/page.md", "a/page.md", "c/page.md", "a/page.md"]

    first = authors_module.make_author_mapping(paths, seed=1)
    second = authors_module.make_author_mapping(reversed(paths), seed=1)

    assert first == second and set(first) == {"a/page.md", "b/page.md", "c/page.md"}
    assert set(first.values()) <= set(authors_module.AUTHORS)
    assert authors_module.make_author_mapping(paths, seed=2) != first


# --- corpus loader against a fake API ---


def uploaded_file(content: bytes) -> tuple[str, bytes]:
    """(file name, file bytes) of the one file part in a multipart body."""
    head, _, rest = content.partition(b'filename="')
    name, _, rest = rest.partition(b'"')
    _, _, rest = rest.partition(b"\r\n\r\n")
    data, _, _ = rest.partition(b"\r\n--")
    return name.decode(), data


class FakeApi:
    """Accepts uploads, reports identical content as `no_changes`, and finishes jobs."""

    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, list[bytes]] = {}
        self.posted: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path.startswith("/jobs/"):
            return httpx.Response(200, json={"status": "ready", "error": None})
        _, data = uploaded_file(request.content)
        self.posted.append({"path": path, "has_index": b'name="index"' in request.content})
        if path == "/documents":
            document_id = uuid.uuid4()
            self.documents[document_id] = [data]
            return httpx.Response(202, json=self._accepted(document_id, 1))
        document_id = uuid.UUID(path.split("/")[2])
        versions = self.documents[document_id]
        if data == versions[-1]:
            return httpx.Response(
                200, json={"status": "no_changes", "document_id": str(document_id)}
            )
        versions.append(data)
        return httpx.Response(202, json=self._accepted(document_id, len(versions)))

    @staticmethod
    def _accepted(document_id: uuid.UUID, version: int) -> dict[str, Any]:
        return {
            "status": "accepted",
            "document_id": str(document_id),
            "version": version,
            "version_id": str(uuid.uuid4()),
            "job_id": str(uuid.uuid4()),
        }


async def test_loader_numbers_versions_by_release_and_reuses_unchanged_pages(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    for release, same_text in (("release-1", "stays"), ("release-2", "stays")):
        (corpus / release / "tasks").mkdir(parents=True)
        (corpus / release / "tasks" / "same.md").write_text(f"# Same\n\n{same_text}\n")
    (corpus / "release-1" / "tasks" / "changes.md").write_text("# Changes\n\nold text\n")
    (corpus / "release-2" / "tasks" / "changes.md").write_text("# Changes\n\nnew text\n")
    assert pages(corpus, "release-1") == ["tasks/changes.md", "tasks/same.md"]
    api = FakeApi()

    async with httpx.AsyncClient(
        base_url="http://api", transport=httpx.MockTransport(api.handler)
    ) as http:
        loader = Loader(http, corpus, "base", {"tasks/same.md": "I. Petrov"}, poll_s=0.0)
        await loader.load_release("release-1")
        await loader.load_release("release-2")
    manifest = loader.manifest(["release-1", "release-2"])

    assert manifest.index == "base"
    assert manifest.version_number("tasks/changes.md", "release-1") == 1
    assert manifest.version_number("tasks/changes.md", "release-2") == 2
    same_1 = manifest.version_number("tasks/same.md", "release-1")
    same_2 = manifest.version_number("tasks/same.md", "release-2")
    assert same_1 == same_2 == 1  # identical content is not a new version
    by_release = {(v.document, v.release): v for v in manifest.versions}
    assert (
        by_release[("tasks/same.md", "release-2")].version_id
        == by_release[("tasks/same.md", "release-1")].version_id
    )
    assert len(manifest.versions) == 4 and all(p["has_index"] for p in api.posted)


def test_cli_parser_knows_every_command() -> None:
    p = parser()
    assert p.parse_args(["metrics", "--run", "x"]).command == "metrics"
    args = p.parse_args(["run", "--config", "c.yaml", "--manifest", "m.json", "--split", "test"])
    assert args.split == "test" and args.repetitions == 2 and args.concurrency == 2
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--config", "c.yaml"])  # the manifest is required


def test_every_committed_eval_config_is_valid() -> None:
    from ragqa.domain.pipeline import PipelineConfig
    from ragqa.evaluation.runner import load_eval_config
    from ragqa.query.service import deep_merge

    files = sorted(Path("eval/configs").glob("*.yaml"))
    assert len(files) >= 8
    for file in files:
        config = load_eval_config(file)
        assert config.name == file.stem
        merged = deep_merge(PipelineConfig(default_model="m").model_dump(), config.pipeline)
        PipelineConfig.model_validate(merged)  # no unknown or invalid fields
