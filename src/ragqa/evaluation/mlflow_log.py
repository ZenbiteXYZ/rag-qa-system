"""Store a run directory in MLflow (ADR-0015, ML_DESIGN section 6).

Uses MLflow's REST API with httpx instead of the `mlflow` client: the app depends on
`mlflow-tracing` only, which has no run logging, and the full client is a heavy dependency for
four calls. The run directory stays the primary record. The MLflow run holds the parameters,
the aggregated metrics and every file of the directory as artifacts, and is never edited after
`log_run` returns.
"""

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ragqa.evaluation.records import RunMeta

_PARAM_LIMIT = 500
_BAD_METRIC_CHARS = re.compile(r"[^A-Za-z0-9_\-. :/]")


def metric_key(name: str) -> str:
    """MLflow metric names allow letters, digits, `_ - . : /` and spaces."""
    return _BAD_METRIC_CHARS.sub("_", name.replace("@", "_at_"))


def flatten(prefix: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for key, item in value.items():
            out |= flatten(f"{prefix}.{key}" if prefix else str(key), item)
        return out
    return {prefix: json.dumps(value) if not isinstance(value, str) else value}


def run_params(meta: RunMeta) -> dict[str, str]:
    params: dict[str, str] = {
        "protocol_version": str(meta.protocol_version),
        "metric_code_version": str(meta.metric_code_version),
        "config_name": meta.name,
        "config_hash": meta.config_hash,
        "split": meta.split,
        "repetitions": str(meta.repetitions),
        "golden_hash": meta.golden_hash,
        "git_sha": meta.git_sha or "",
        "uv_lock_hash": meta.uv_lock_hash or "",
        "prompt_version": meta.prompt_version or "",
        "index": json.dumps(meta.index),
    }
    params |= flatten("pipeline", meta.pipeline_config)
    params |= flatten("pins", meta.models)
    return {k: v[:_PARAM_LIMIT] for k, v in params.items() if v != ""}


class MlflowRest:
    def __init__(self, http: httpx.Client, base_url: str) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        reply = self._http.post(f"{self._base}/api/2.0/mlflow/{path}", json=body)
        reply.raise_for_status()
        data: dict[str, Any] = reply.json()
        return data

    def experiment_id(self, name: str) -> str:
        reply = self._http.get(
            f"{self._base}/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name}
        )
        if reply.status_code == 200:
            return str(reply.json()["experiment"]["experiment_id"])
        return str(self._post("experiments/create", {"name": name})["experiment_id"])

    def create_run(self, experiment_id: str, name: str, tags: dict[str, str]) -> dict[str, Any]:
        body = {
            "experiment_id": experiment_id,
            "run_name": name,
            "start_time": int(time.time() * 1000),
            "tags": [{"key": k, "value": v} for k, v in tags.items()],
        }
        run: dict[str, Any] = self._post("runs/create", body)["run"]
        return run

    def log_batch(self, run_id: str, params: dict[str, str], metrics: dict[str, float]) -> None:
        stamp = int(time.time() * 1000)
        items = list(params.items())
        pairs = list(metrics.items())
        # The API takes at most 100 params and 1000 metrics per call.
        for start in range(0, max(len(items), 1), 100):
            self._post(
                "runs/log-batch",
                {
                    "run_id": run_id,
                    "params": [{"key": k, "value": v} for k, v in items[start : start + 100]],
                },
            )
        for start in range(0, len(pairs), 1000):
            self._post(
                "runs/log-batch",
                {
                    "run_id": run_id,
                    "metrics": [
                        {"key": k, "value": v, "timestamp": stamp, "step": 0}
                        for k, v in pairs[start : start + 1000]
                    ],
                },
            )

    def upload(self, artifact_root: str, files: list[Path], base: Path) -> None:
        """`artifact_root` is the run's `mlflow-artifacts:/<experiment>/<run>/artifacts` URI,
        written to through the server's artifact proxy."""
        root = artifact_root.removeprefix("mlflow-artifacts:/").strip("/")
        for file in files:
            relative = file.relative_to(base).as_posix()
            reply = self._http.put(
                f"{self._base}/api/2.0/mlflow-artifacts/artifacts/{root}/{relative}",
                content=file.read_bytes(),
            )
            reply.raise_for_status()

    def finish(self, run_id: str) -> None:
        self._post(
            "runs/update",
            {"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)},
        )


def log_run(
    run_dir: Path,
    *,
    tracking_uri: str,
    experiment: str,
    meta: RunMeta,
    aggregates: dict[str, float],
    http: httpx.Client | None = None,
) -> str:
    """Create the MLflow run for a run directory and return its id."""
    client = MlflowRest(http or httpx.Client(timeout=60.0), tracking_uri)
    tags = {
        "protocol_version": str(meta.protocol_version),
        "git_dirty": str(meta.git_dirty),
        "run_dir": run_dir.name,
        "mlflow.runName": meta.run_id,
    }
    if meta.recomputed_from:
        tags["recomputed_from"] = meta.recomputed_from
    run = client.create_run(client.experiment_id(experiment), meta.run_id, tags)
    run_id = run["info"]["run_id"]
    client.log_batch(run_id, run_params(meta), {metric_key(k): v for k, v in aggregates.items()})
    files = sorted(p for p in run_dir.rglob("*") if p.is_file())
    client.upload(run["info"]["artifact_uri"], files, run_dir)
    client.finish(run_id)
    return str(run_id)
