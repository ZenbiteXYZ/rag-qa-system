import json
from typing import Any

import pytest

from ragqa.domain.pipeline import PipelineConfig, PipelineState
from ragqa.evaluation.compare import bootstrap_mean_ci, paired_comparison
from ragqa.evaluation.judge import (
    CORRECTNESS,
    FAITHFULNESS,
    RELEVANCY,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    RelevancyVerdict,
    correctness_score,
    faithfulness_score,
    judge_jobs,
    relevancy_score,
    run_jobs,
)
from ragqa.evaluation.records import QuestionRecord
from ragqa.llm.types import LLMUnavailable
from tests.query_fakes import INDEX, FakeLLM, chunk


def per_question(values: dict[str, float | None]) -> dict[str, dict[str, float | None]]:
    return {k: {"m": v} for k, v in values.items()}


def test_paired_comparison_counts_and_interval() -> None:
    base = per_question({"a": 0.2, "b": 0.5, "c": 0.5, "d": 0.4, "e": None})
    other = per_question({"a": 0.6, "b": 0.9, "c": 0.5, "d": 0.8, "f": 1.0})

    result = paired_comparison(base, other, "m", seed=1, resamples=2000)

    assert result is not None
    assert (result.n, result.better, result.worse, result.unchanged) == (4, 3, 0, 1)
    assert result.mean_diff == pytest.approx(0.3)
    assert result.ci_low > 0 and result.distinguishable
    assert result.excluded == 1  # "f" has a value only in the other run; "e" in neither


def test_a_difference_inside_the_noise_is_not_distinguishable() -> None:
    base = per_question({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5})
    other = per_question({"a": 0.6, "b": 0.6, "c": 0.6, "d": 0.6})

    result = paired_comparison(base, other, "m", noise=0.3, seed=1, resamples=500)

    assert result is not None and result.ci_low > 0
    assert result.distinguishable is False  # 0.1 is smaller than the run-to-run noise of 0.3


def test_an_interval_that_contains_zero_is_not_distinguishable() -> None:
    base = per_question({"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0})
    other = per_question({"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0})

    result = paired_comparison(base, other, "m", seed=1, resamples=500)

    assert result is not None and result.mean_diff == 0.0 and not result.distinguishable


def test_no_common_values_gives_none() -> None:
    assert paired_comparison(per_question({"a": None}), per_question({"a": 1.0}), "m") is None


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    diffs = [0.1, 0.2, -0.1, 0.4]
    assert bootstrap_mean_ci(diffs, seed=3, resamples=300) == bootstrap_mean_ci(
        diffs, seed=3, resamples=300
    )


def test_verdict_scores() -> None:
    assert [
        correctness_score(CorrectnessVerdict(verdict=v))
        for v in ("correct", "partly_correct", "wrong")
    ] == [1.0, 0.5, 0.0]
    assert faithfulness_score(FaithfulnessVerdict(claims=[])) is None
    verdict = FaithfulnessVerdict.model_validate(
        {"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]}
    )
    assert faithfulness_score(verdict) == 0.5
    assert [relevancy_score(RelevancyVerdict(score=s)) for s in (1, 3, 5)] == [0.0, 0.5, 1.0]


def _record() -> QuestionRecord:
    return QuestionRecord(
        question_id="q1",
        repetition=0,
        split="dev",
        category="single_section",
        request={},
        client_latency_ms=1.0,
        started_at="t",
    )


def _state(**update: Any) -> PipelineState:
    update.setdefault("config", PipelineConfig(default_model="m"))
    return PipelineState(question="q", index=INDEX, answer="Use drain [1].", **update)


def test_a_grounded_answer_gets_three_judgments_a_refusal_one() -> None:
    grounded = _state(context_chunks=(chunk(0),))
    assert [j[0] for j in judge_jobs(_record(), grounded, "q", "ref")] == [
        CORRECTNESS,
        RELEVANCY,
        FAITHFULNESS,
    ]
    refusal = _state(abstained=True)
    assert [j[0] for j in judge_jobs(_record(), refusal, "q", "ref")] == [CORRECTNESS]
    closed_book = _state(
        config=PipelineConfig.model_validate({"default_model": "m", "retrieve": {"enabled": False}})
    )
    assert [j[0] for j in judge_jobs(_record(), closed_book, "q", "ref")] == [
        CORRECTNESS,
        RELEVANCY,
    ]


async def test_judge_calls_use_the_pinned_settings_and_store_raw_output() -> None:
    llm = FakeLLM(structured=[json.dumps({"verdict": "partly_correct", "reason": "misses X"})])
    jobs = [
        (("q1", 0), name, messages, schema, score)
        for name, messages, schema, score in judge_jobs(
            _record(), _state(abstained=True), "q", "ref"
        )
    ]

    (record,) = await run_jobs(llm, jobs, model="gpt-6-luna", seed=7)

    assert record.metric == CORRECTNESS and record.score == 0.5
    assert record.judge_model == "gpt-6-luna" and record.seed == 7
    assert record.resolved_model == "provider/gpt-6-luna" and record.rubric_version == "1"
    assert record.verdict == {"verdict": "partly_correct", "reason": "misses X"}
    assert record.raw_text is not None and record.prompt_tokens == 100
    assert llm.calls[0]["model"] == "gpt-6-luna"


async def test_a_failed_judge_call_is_recorded_not_dropped() -> None:
    llm = FakeLLM(structured=["{}"], fail=LLMUnavailable("down"))
    jobs = [
        (("q1", 0), name, messages, schema, score)
        for name, messages, schema, score in judge_jobs(
            _record(), _state(abstained=True), "q", "ref"
        )
    ]

    (record,) = await run_jobs(llm, jobs, model="j", seed=1)

    assert record.score is None and record.error is not None and "LLMUnavailable" in record.error
