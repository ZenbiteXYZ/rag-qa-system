"""Runs the fixed stage sequence for one question (ADR-0010)."""

from collections.abc import Sequence

from ragqa.domain.pipeline import PipelineState
from ragqa.query.stages.base import Stage


async def run_pipeline(stages: Sequence[Stage], state: PipelineState) -> PipelineState:
    """Run each enabled stage in order inside `traced_stage`, and stop once `state.outcome`
    is set (a clarification or a deleted document ends the run early).

    TODO(author)
    """
    raise NotImplementedError
