"""Runs the fixed stage sequence for one question (ADR-0010)."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ragqa.domain.pipeline import PipelineState, StageTiming
from ragqa.ingestion.chunkers import TokenCounterFactory
from ragqa.llm.types import LLMClient
from ragqa.observability.logging import get_logger
from ragqa.observability.tracing import traced_stage
from ragqa.query.reranker import Reranker
from ragqa.query.retriever import Retriever
from ragqa.query.stages.base import BOOKKEEPING, Stage
from ragqa.query.stages.context import ContextStage
from ragqa.query.stages.generate import GenerateStage
from ragqa.query.stages.hyde import HydeStage
from ragqa.query.stages.rerank import RerankStage
from ragqa.query.stages.retrieve import RetrieveStage
from ragqa.query.stages.rewrite import RewriteStage
from ragqa.query.stages.self_query import KnownValuesProvider, SelfQueryStage

log = get_logger(__name__)


class StageContractError(Exception):
    """A stage changed a state field that it does not declare in `writes`. A bug."""


@dataclass(frozen=True)
class QueryDeps:
    llm: LLMClient
    retriever: Retriever
    known_values: KnownValuesProvider
    token_counter: TokenCounterFactory | None
    reranker: Reranker | None = None


def build_stages(deps: QueryDeps) -> list[Stage]:
    """The stages in order: rewrite, Self-Query, HyDE, retrieve, rerank, context, generate."""
    stages: list[Stage] = [
        RewriteStage(deps.llm),
        SelfQueryStage(deps.llm, deps.known_values),
        HydeStage(deps.llm),
        RetrieveStage(deps.retriever),
    ]
    if deps.reranker is not None:
        stages.append(RerankStage(deps.reranker))
    stages += [ContextStage(deps.retriever, deps.token_counter), GenerateStage(deps.llm)]
    return stages


def _check_writes(stage: Stage, before: PipelineState, after: PipelineState) -> None:
    allowed = set(stage.writes) | BOOKKEEPING
    for field in PipelineState.model_fields:
        if field not in allowed and getattr(before, field) != getattr(after, field):
            raise StageContractError(f"stage {stage.name} changed {field}, which it does not write")


def _dump(state: PipelineState, fields: Sequence[str]) -> dict[str, object]:
    return state.model_dump(mode="json", include=set(fields))


async def run_stage(stage: Stage, state: PipelineState) -> PipelineState:
    start = time.perf_counter()
    try:
        with traced_stage(stage.name, reads=stage.reads, writes=stage.writes) as span:
            span.set_inputs(_dump(state, stage.reads))
            new = await stage.run(state)
            _check_writes(stage, state, new)
            span.set_outputs(_dump(new, stage.writes))
    except Exception as exc:
        if not stage.optional:
            raise
        log.warning("stage_degraded", stage=stage.name, error_type=type(exc).__name__, exc_info=exc)
        note = f"{stage.name} failed and was skipped: {type(exc).__name__}"
        new = state.model_copy(
            update={
                "warnings": (*state.warnings, note),
                "degraded": (*state.degraded, stage.name),
            }
        )
    ms = round((time.perf_counter() - start) * 1000, 1)
    return new.model_copy(update={"timings": (*new.timings, StageTiming(stage=stage.name, ms=ms))})


async def run_pipeline(stages: Sequence[Stage], state: PipelineState) -> PipelineState:
    """Run each enabled stage in order inside `traced_stage`, and stop once `state.outcome`
    is set (a clarification or a deleted document ends the run early).

    An optional stage that fails is skipped and recorded in `state.degraded`, so the request
    still gets an answer; a required stage that fails fails the request. Nothing is dropped
    silently: the evaluation counts degraded stages (ML_DESIGN section 6, rule 3).
    """
    for stage in stages:
        if state.outcome is not None:
            break
        if stage.enabled(state.config):
            state = await run_stage(stage, state)
    return state
