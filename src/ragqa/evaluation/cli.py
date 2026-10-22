"""Command line of the evaluation: `python -m ragqa.evaluation <command>`.

Typical order: `validate-golden`, `calibrate` (once per judge and rubric), then per configuration
`run`, `judge`, `metrics`, and `compare` against the baseline run, optionally `mlflow`.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from ragqa.evaluation.calibrate import calibrate, calibration_ok, load_items
from ragqa.evaluation.golden import GoldenSnapshot, load_golden
from ragqa.evaluation.judge import RUBRIC_VERSION, judge_jobs, run_jobs
from ragqa.evaluation.metrics import state_of
from ragqa.evaluation.mlflow_log import log_run
from ragqa.evaluation.records import METRIC_CODE_VERSION, RunDir, RunMeta, append_jsonl, load_run
from ragqa.evaluation.report import build_metrics, dump, render_comparison, render_report
from ragqa.evaluation.runner import create_run
from ragqa.llm.client import ProxyLLMClient

DEFAULT_SEED = 7


def llm_client(url: str) -> ProxyLLMClient:
    # The key is read from the environment and never printed.
    key = os.environ.get("RAGQA_LITELLM_API_KEY", "unset")
    return ProxyLLMClient(AsyncOpenAI(base_url=url.rstrip("/") + "/v1", api_key=key, timeout=120.0))


def snapshot_of(path: Path, corpus: Path | None) -> tuple[Any, GoldenSnapshot]:
    from ragqa.evaluation.text_sources import corpus_text

    golden = load_golden(path)
    if corpus is not None:
        os.environ["RAGQA_CORPUS_DIR"] = str(corpus)
    return golden, golden.snapshot(corpus_text)


def cmd_validate_golden(args: argparse.Namespace) -> int:
    golden = load_golden(args.golden)
    issues = golden.check_composition()
    try:
        snapshot_of(args.golden, args.corpus)
    except ValueError as exc:
        issues.append(str(exc))
    for issue in issues:
        print(issue)
    print(f"{len(golden.questions)} questions, golden set {golden.hash[:12]}, {len(issues)} issues")
    return 1 if issues else 0


async def cmd_run(args: argparse.Namespace) -> int:
    golden, snapshot = snapshot_of(args.golden, args.corpus)
    done = 0

    def progress(record: Any) -> None:
        nonlocal done
        done += 1
        status = "error" if record.error else "ok"
        print(f"  {done}: {record.question_id} rep {record.repetition} {status}", file=sys.stderr)

    path = await create_run(
        api_url=args.api,
        golden=golden,
        snapshot=snapshot,
        config_path=args.config,
        manifest_path=args.manifest,
        out=args.out,
        split=args.split,
        repetitions=args.repetitions,
        concurrency=args.concurrency,
        progress=progress,
    )
    print(path)
    return 0


async def cmd_judge(args: argparse.Namespace) -> int:
    ok, reason = calibration_ok(args.calibration, model=args.model)
    if not ok and not args.uncalibrated:
        print(f"refusing to judge: {reason} (ADR-0019). Use --uncalibrated to override.")
        return 1
    run_data, run = load_run(args.run), RunDir(args.run)
    questions = run_data.golden.by_id()
    done = {(j.question_id, j.repetition, j.metric) for j in run_data.judge}
    jobs = []
    for record in run_data.records:
        state = state_of(record)
        if state is None or state.answer is None:
            continue
        q = questions[record.question_id].question
        for metric, messages, schema, score_of in judge_jobs(
            record, state, q.question, q.reference_answer
        ):
            if (record.question_id, record.repetition, metric) not in done:
                jobs.append(
                    ((record.question_id, record.repetition), metric, messages, schema, score_of)
                )
    records = await run_jobs(
        llm_client(args.llm_url),
        jobs,
        model=args.model,
        seed=args.seed,
        concurrency=args.concurrency,
    )
    for judged in records:
        append_jsonl(run.judge, judged)
    meta = {
        "judge_model": args.model,
        "rubric_version": RUBRIC_VERSION,
        "calibrated": ok,
        "reason": reason,
    }
    (args.run / "judge.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    failed = sum(r.error is not None for r in records)
    print(f"{len(records)} judgments written, {failed} failed")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    run_data, run = load_run(args.run), RunDir(args.run)
    metrics = build_metrics(run_data)
    protocol, code = metrics["protocol_version"], METRIC_CODE_VERSION
    report = render_report(metrics)
    judge_meta = args.run / "judge.meta.json"
    if judge_meta.exists() and not json.loads(judge_meta.read_text())["calibrated"]:
        report = (
            "> The judge was not calibrated (ADR-0019): judged scores are not usable.\n\n" + report
        )
    run.write_once(run.metrics(protocol, code), dump(metrics))
    run.write_once(run.report(protocol, code), report)
    print(run.report(protocol, code))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    def load(run_dir: Path) -> dict[str, Any]:
        files = sorted(run_dir.glob("metrics.p*.m*.json"))
        if not files:
            raise SystemExit(f"{run_dir}: run `metrics` first")
        data: dict[str, Any] = json.loads(files[-1].read_text(encoding="utf-8"))
        return data

    try:
        text = render_comparison(load(args.base), load(args.other))
    except ValueError as exc:
        print(f"not comparable: {exc}")
        return 1
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


async def cmd_calibrate(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"{args.out} exists; a calibration result is a record, choose another file")
        return 1
    result = await calibrate(
        llm_client(args.llm_url), load_items(args.items), model=args.model, seed=args.seed
    )
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for n, p in enumerate(result["passes"], start=1):
        wrong = p["wrong_judged_correct"]
        print(f"pass {n}: agreement {p['agreement']:.2f}, wrong judged correct {wrong}")
    print(f"self agreement {result['self_agreement']}, gate passed: {result['gate_passed']}")
    for note in result["notes"]:
        print(f"note: {note}")
    return 0 if result["gate_passed"] else 1


def cmd_mlflow(args: argparse.Namespace) -> int:
    run_data = load_run(args.run)
    files = sorted(args.run.glob("metrics.p*.m*.json"))
    if not files:
        print("run `metrics` first")
        return 1
    metrics = json.loads(files[-1].read_text(encoding="utf-8"))
    aggregates = {n: a["mean"] for n, a in metrics["overall"].items() if a["mean"] is not None}
    run_id = log_run(
        args.run,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        meta=RunMeta.model_validate(run_data.meta.model_dump()),
        aggregates=aggregates,
    )
    print(f"MLflow run {run_id}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ragqa.evaluation", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, **kw: Any) -> argparse.ArgumentParser:
        return sub.add_parser(name, **kw)

    v = add("validate-golden", help="check the golden set against ADR-0019 and the corpus")
    v.add_argument("golden", type=Path)
    v.add_argument("--corpus", type=Path)

    r = add("run", help="send the questions to the API and store a run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--golden", type=Path, default=Path("eval/golden"))
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--corpus", type=Path)
    r.add_argument("--api", default="http://localhost:8000")
    r.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    r.add_argument("--repetitions", type=int, default=2)
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--out", type=Path, default=Path("eval/reports"))

    j = add("judge", help="grade the answers of a run with the LLM judge")
    j.add_argument("--run", type=Path, required=True)
    j.add_argument("--model", default="gpt-6-luna")
    j.add_argument("--llm-url", default="http://localhost:4000")
    j.add_argument("--seed", type=int, default=DEFAULT_SEED)
    j.add_argument("--concurrency", type=int, default=4)
    j.add_argument("--calibration", type=Path, default=Path("eval/reports/calibration.json"))
    j.add_argument("--uncalibrated", action="store_true")

    m = add("metrics", help="compute metrics and the report of a run, offline")
    m.add_argument("--run", type=Path, required=True)

    c = add("compare", help="paired comparison of two runs")
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--other", type=Path, required=True)
    c.add_argument("--out", type=Path)

    k = add("calibrate", help="the judge calibration gate")
    k.add_argument("--items", type=Path, default=Path("eval/golden/calibration.yaml"))
    k.add_argument("--model", default="gpt-6-luna")
    k.add_argument("--llm-url", default="http://localhost:4000")
    k.add_argument("--seed", type=int, default=DEFAULT_SEED)
    k.add_argument("--out", type=Path, default=Path("eval/reports/calibration.json"))

    f = add("mlflow", help="store a run directory in MLflow")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--tracking-uri", default="http://localhost:5000")
    f.add_argument("--experiment", default="ragqa-eval")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    sync = {
        "validate-golden": cmd_validate_golden,
        "metrics": cmd_metrics,
        "compare": cmd_compare,
        "mlflow": cmd_mlflow,
    }
    asynchronous = {"run": cmd_run, "judge": cmd_judge, "calibrate": cmd_calibrate}
    if args.command in sync:
        return sync[args.command](args)
    return asyncio.run(asynchronous[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
