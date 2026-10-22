"""The LLM judge (ADR-0019): rubrics, structured verdicts and scoring.

The judge is a different model family from the generator, called through the same proxy with
structured output, a low reasoning effort and a seed. Its inputs and raw outputs are stored in
`judge.jsonl`, so scores can be recomputed. Any change to a rubric text, the judge model or the
scoring is a protocol change: bump `RUBRIC_VERSION` and create protocol v2 (ADR-0019).
"""

import asyncio
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from ragqa.domain.conversation import ChatMessage
from ragqa.domain.pipeline import PipelineState
from ragqa.evaluation.records import JudgeRecord, QuestionRecord
from ragqa.llm.types import LLMClient, LLMError

RUBRIC_VERSION = "1"
JUDGE_REASONING_EFFORT = "low"
JUDGE_MAX_TOKENS = 1200

CORRECTNESS = "answer_correctness"
FAITHFULNESS = "faithfulness"
RELEVANCY = "answer_relevancy"

CORRECTNESS_SCORES = {"correct": 1.0, "partly_correct": 0.5, "wrong": 0.0}

CORRECTNESS_SYSTEM = """You grade an answer to a question about technical documentation by \
comparing it with a reference answer written by an expert.
Judge only agreement with the reference:
- "correct": the facts of the answer match the reference, nothing in it contradicts the \
reference, and nothing essential from the reference is missing. Different wording, extra \
correct detail and a different order are fine.
- "partly_correct": the answer is right on some essential points but misses an essential point \
or adds a statement that contradicts the reference on a minor point.
- "wrong": the answer contradicts the reference on an essential point, answers a different \
question, or states nothing that matches the reference.
If the reference says the documents do not contain the answer, an answer that declines to \
answer is "correct" and an answer that states facts is "wrong".
Fluent, confident wording is not evidence of correctness. Return your verdict and a one \
sentence reason."""

FAITHFULNESS_SYSTEM = """You check an answer against the excerpts it was written from.
Split the answer into its individual factual claims. For each claim decide whether the \
excerpts support it: true if an excerpt states it or it follows directly from an excerpt, false \
otherwise. Judge only against the excerpts, not against what you know. Ignore citation \
markers like [1] and statements that only restate the question. Return every claim with your \
decision."""

RELEVANCY_SYSTEM = """You rate how directly an answer addresses a question, from 1 to 5. You \
do not know the correct answer and must not use any reference.
5: addresses exactly what was asked, nothing missing, no padding.
4: addresses the question with minor padding or a minor gap.
3: partly addresses it, or contains a lot of unrelated material.
2: mostly off topic, or only touches the question.
1: does not address the question.
Return the score and a one sentence reason."""


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partly_correct", "wrong"]
    reason: str = ""


class Claim(BaseModel):
    claim: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class RelevancyVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = ""


def correctness_messages(question: str, reference: str, answer: str) -> list[ChatMessage]:
    user = f"Question:\n{question}\n\nReference answer:\n{reference}\n\nAnswer to grade:\n{answer}"
    return [
        ChatMessage(role="system", content=CORRECTNESS_SYSTEM),
        ChatMessage(role="user", content=user),
    ]


def faithfulness_messages(context: Sequence[str], answer: str) -> list[ChatMessage]:
    excerpts = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(context, start=1))
    user = f"Excerpts:\n\n{excerpts}\n\nAnswer:\n{answer}"
    return [
        ChatMessage(role="system", content=FAITHFULNESS_SYSTEM),
        ChatMessage(role="user", content=user),
    ]


def relevancy_messages(question: str, answer: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=RELEVANCY_SYSTEM),
        ChatMessage(role="user", content=f"Question:\n{question}\n\nAnswer:\n{answer}"),
    ]


def correctness_score(verdict: CorrectnessVerdict) -> float:
    return CORRECTNESS_SCORES[verdict.verdict]


def faithfulness_score(verdict: FaithfulnessVerdict) -> float | None:
    """Share of supported claims. None when the answer makes no claims."""
    if not verdict.claims:
        return None
    return sum(c.supported for c in verdict.claims) / len(verdict.claims)


def relevancy_score(verdict: RelevancyVerdict) -> float:
    return (verdict.score - 1) / 4


async def judge_one(
    llm: LLMClient,
    *,
    key: tuple[str, int],
    metric: str,
    messages: list[ChatMessage],
    schema: type[BaseModel],
    score_of: object,
    model: str,
    seed: int,
) -> JudgeRecord:
    question_id, repetition = key
    base = {
        "question_id": question_id,
        "repetition": repetition,
        "metric": metric,
        "judge_model": model,
        "rubric_version": RUBRIC_VERSION,
        "seed": seed,
    }
    try:
        verdict, completion = await llm.complete_structured(
            messages,
            schema,
            model=model,
            max_tokens=JUDGE_MAX_TOKENS,
            reasoning_effort=JUDGE_REASONING_EFFORT,
            seed=seed,
        )
    except LLMError as exc:
        return JudgeRecord(**base, error=f"{type(exc).__name__}: {exc}")  # type: ignore[arg-type]
    usage = completion.usage
    return JudgeRecord(
        **base,  # type: ignore[arg-type]
        resolved_model=completion.resolved_model,
        verdict=verdict.model_dump(mode="json"),
        score=score_of(verdict),  # type: ignore[operator]
        raw_text=completion.text,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
    )


def judge_jobs(
    record: QuestionRecord, state: PipelineState, question: str, reference: str
) -> list[tuple[str, list[ChatMessage], type[BaseModel], object]]:
    """The judgments one answered request needs: correctness always, faithfulness when the
    answer was written from excerpts, relevancy for real (non-refusal) answers."""
    answer = state.answer or ""
    jobs: list[tuple[str, list[ChatMessage], type[BaseModel], object]] = [
        (
            CORRECTNESS,
            correctness_messages(question, reference, answer),
            CorrectnessVerdict,
            correctness_score,
        )
    ]
    if not state.abstained:
        jobs.append(
            (RELEVANCY, relevancy_messages(question, answer), RelevancyVerdict, relevancy_score)
        )
        if state.config.retrieve.enabled and state.context_chunks:
            context = [c.text for c in state.context_chunks]
            jobs.append(
                (
                    FAITHFULNESS,
                    faithfulness_messages(context, answer),
                    FaithfulnessVerdict,
                    faithfulness_score,
                )
            )
    return jobs


async def run_jobs(
    llm: LLMClient,
    jobs: list[tuple[tuple[str, int], str, list[ChatMessage], type[BaseModel], object]],
    *,
    model: str,
    seed: int,
    concurrency: int = 4,
) -> list[JudgeRecord]:
    gate = asyncio.Semaphore(concurrency)

    async def one(
        job: tuple[tuple[str, int], str, list[ChatMessage], type[BaseModel], object],
    ) -> JudgeRecord:
        key, metric, messages, schema, score_of = job
        async with gate:
            return await judge_one(
                llm,
                key=key,
                metric=metric,
                messages=messages,
                schema=schema,
                score_of=score_of,
                model=model,
                seed=seed,
            )

    return list(await asyncio.gather(*(one(j) for j in jobs)))
