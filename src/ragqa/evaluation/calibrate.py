"""The judge calibration gate (ADR-0019, ML_DESIGN section 4).

Before the judge's scores are used, it grades about 30 hand-labeled dev answers twice. The gate,
fixed before looking at any result: exact agreement with the hand labels of at least 80% in
each pass, and at most 10% of the answers labeled "wrong" judged "correct". A new judge or a
new rubric is a protocol change and has to pass again.

Items file (YAML list, written by hand):

    - id: c01
      question: How do I drain a node?
      reference_answer: Run kubectl drain with --ignore-daemonsets.
      answer: ...                       # a generated answer from a dev run
      label: correct                    # correct, partly_correct or wrong
"""

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from ragqa.evaluation.judge import (
    CORRECTNESS,
    RUBRIC_VERSION,
    CorrectnessVerdict,
    correctness_messages,
    correctness_score,
    judge_one,
)
from ragqa.llm.types import LLMClient

MIN_AGREEMENT = 0.80
MAX_WRONG_JUDGED_CORRECT = 0.10
EXPECTED_ITEMS = 30
PASSES = 2

Label = Literal["correct", "partly_correct", "wrong"]


class CalibrationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    label: Label


def load_items(path: Path) -> list[CalibrationItem]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [CalibrationItem.model_validate(item) for item in raw]


def evaluate(
    items: Sequence[CalibrationItem], passes: Sequence[Sequence[str | None]]
) -> dict[str, Any]:
    """The gate numbers from the verdicts of each pass (None where the judge call failed)."""
    wrong = [i for i, item in enumerate(items) if item.label == "wrong"]
    per_pass: list[dict[str, float | None]] = []
    passed = bool(passes)
    for verdicts in passes:
        agreement = sum(v == item.label for v, item in zip(verdicts, items, strict=True)) / len(
            items
        )
        wrong_rate = sum(verdicts[i] == "correct" for i in wrong) / len(wrong) if wrong else None
        per_pass.append({"agreement": agreement, "wrong_judged_correct": wrong_rate})
        passed = passed and (
            agreement >= MIN_AGREEMENT
            and wrong_rate is not None
            and wrong_rate <= MAX_WRONG_JUDGED_CORRECT
        )
    self_agreement = (
        sum(a == b for a, b in zip(passes[0], passes[1], strict=True)) / len(items)
        if len(passes) >= 2
        else None
    )
    notes = []
    if len(items) != EXPECTED_ITEMS:
        notes.append(f"{len(items)} items, the protocol asks for {EXPECTED_ITEMS}")
    if not wrong:
        notes.append("no answer is labeled wrong, so the gate cannot pass")
    return {
        "items": len(items),
        "passes": per_pass,
        "self_agreement": self_agreement,
        "gate_passed": passed,
        "notes": notes,
    }


async def calibrate(
    llm: LLMClient, items: Sequence[CalibrationItem], *, model: str, seed: int
) -> dict[str, Any]:
    async def one(item: CalibrationItem, pass_number: int) -> str | None:
        record = await judge_one(
            llm,
            key=(item.id, pass_number),
            metric=CORRECTNESS,
            messages=correctness_messages(item.question, item.reference_answer, item.answer),
            schema=CorrectnessVerdict,
            score_of=correctness_score,
            model=model,
            seed=seed,
        )
        return record.verdict["verdict"] if record.verdict else None

    passes = [list(await asyncio.gather(*(one(item, n) for item in items))) for n in range(PASSES)]
    result = evaluate(items, passes)
    result |= {
        "judge_model": model,
        "rubric_version": RUBRIC_VERSION,
        "seed": seed,
        "calibrated_at": datetime.now(UTC).isoformat(),
        "verdicts": [
            {"id": item.id, "label": item.label, "passes": [p[i] for p in passes]}
            for i, item in enumerate(items)
        ],
    }
    return result


def calibration_ok(path: Path, *, model: str) -> tuple[bool, str]:
    """Whether a stored calibration result lets `model` with the current rubric be used."""
    if not path.exists():
        return False, f"no calibration result at {path}"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("judge_model") != model or result.get("rubric_version") != RUBRIC_VERSION:
        return False, "the calibration is for another judge model or rubric version"
    if not result.get("gate_passed"):
        return False, "the calibration did not pass the gate"
    return True, "calibrated"
