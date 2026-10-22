"""The evaluation chain against the real API, Postgres and Qdrant (fake embedder and LLM).

It proves what the unit tests cannot: that the spans of chunks stored by real ingestion and the
spans of evidence quotes resolved from the parser output agree, so coverage is computed on the
same characters.
"""

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from ragqa.evaluation.golden import load_golden
from ragqa.evaluation.manifest import IngestManifest, ManifestVersion
from ragqa.evaluation.records import load_run
from ragqa.evaluation.report import build_metrics, render_report
from ragqa.evaluation.runner import create_run
from ragqa.ingestion.parsers import default_registry
from tests.integration.conftest import POSTGRES_DSN, QDRANT_URL
from tests.integration.test_api_flow import (
    DRAIN,
    RBAC,
    _create_index,
    _ingest_all,
    _qa,
    _upload,
)
from tests.query_fakes import FakeLLM

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        POSTGRES_DSN is None or QDRANT_URL is None,
        reason="RAGQA_TEST_POSTGRES_DSN and RAGQA_TEST_QDRANT_URL are not both set",
    ),
]

RELEASE = "release-1"
GOLDEN = f"""
- id: drain-1
  category: single_section
  split: dev
  question: Use kubectl drain to empty a node.
  reference_answer: Use kubectl drain.
  evidence:
    - {{document: tasks/drain.md, version: {RELEASE}, quote: "Use kubectl drain to empty a node."}}
- id: rbac-1
  category: single_section
  split: dev
  question: Roles grant permissions to subjects.
  reference_answer: Roles grant permissions.
  evidence:
    - {{document: tasks/rbac.md, version: {RELEASE}, quote: "Roles grant permissions to subjects."}}
- id: none-1
  category: unanswerable
  split: dev
  question: How do I train a model?
  reference_answer: The documents do not say.
"""


async def test_run_metrics_on_really_ingested_documents(
    app: FastAPI, client: httpx.AsyncClient, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus" / RELEASE / "tasks"
    corpus.mkdir(parents=True)
    (corpus / "drain.md").write_bytes(DRAIN)
    (corpus / "rbac.md").write_bytes(RBAC)

    await _create_index(client)
    uploaded = {
        "tasks/drain.md": await _upload(client, "drain.md", DRAIN, "Drain a node", "I. Petrov"),
        "tasks/rbac.md": await _upload(client, "rbac.md", RBAC, "RBAC", "A. Ivanova"),
    }
    await _ingest_all(app)
    # Answers cite the first excerpt; the refusal sentence is only used for the unanswerable one.
    _qa(app, FakeLLM(texts=["It says so [1]."]))
    manifest = IngestManifest(
        index="base",
        corpus_releases=(RELEASE,),
        versions=tuple(
            ManifestVersion(
                document=path,
                release=RELEASE,
                document_id=body["document_id"],
                version_id=body["version_id"],
                version=body["version"],
            )
            for path, body in uploaded.items()
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    golden_path = tmp_path / "golden.yaml"
    golden_path.write_text(GOLDEN, encoding="utf-8")
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("name: baseline\nindex: base\npipeline: {}\n", encoding="utf-8")
    parsers = default_registry()

    def corpus_text(document: str, version: str) -> str:
        path = tmp_path / "corpus" / version / document
        return parsers.for_filename(path.name).parse(path.read_bytes(), filename=path.name).text

    golden = load_golden(golden_path)
    run_path = await create_run(
        api_url="http://test",
        golden=golden,
        snapshot=golden.snapshot(corpus_text),
        config_path=config_path,
        manifest_path=manifest_path,
        out=tmp_path / "reports",
        split="dev",
        repetitions=2,
        transport=httpx.ASGITransport(app=app),
    )

    run = load_run(run_path)
    assert len(run.records) == 6 and all(r.error is None for r in run.records)
    metrics = build_metrics(run)
    overall = metrics["overall"]
    # The query text equals the evidence paragraph, so dense retrieval returns that chunk, and
    # its stored span must lie exactly on the evidence span the quote resolved to.
    assert overall["coverage@3"]["n"] == 2 and overall["coverage@3"]["mean"] == 1.0
    assert overall["recall@3@0.8"]["mean"] == 1.0 and overall["mrr"]["mean"] == 1.0
    assert overall["containment@3"]["mean"] == 1.0
    assert (
        overall["citation_support"]["mean"] == 1.0 and overall["citation_validity"]["mean"] == 1.0
    )
    assert overall["context_coverage"]["mean"] == 1.0
    # Nothing was retrieved for the unanswerable question's chunks that matches evidence, and the
    # fake LLM answers it instead of refusing.
    assert overall["abstention_recall"]["mean"] == 0.0
    assert metrics["errors"] == [] and metrics["latency"]["stages"]["retrieve"]["n"] == 6
    assert "coverage@3" in render_report(metrics)
    assert json.loads((run_path / "finished.json").read_text())["errors"] == 0
