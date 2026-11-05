"""Reranking: a cross-encoder reorders the candidates (ADR-0023)."""

from ragqa.domain.pipeline import PipelineConfig, PipelineState
from ragqa.query.reranker import Reranker
from ragqa.query.stages.common import base_query


class RerankStage:
    name = "rerank"
    reads = ("question", "rewritten_query", "semantic_query", "candidates")
    writes = ("ranked", "reranker")
    optional = True

    def __init__(self, reranker: Reranker) -> None:
        self._reranker = reranker

    def enabled(self, config: PipelineConfig) -> bool:
        return config.rerank.enabled

    async def run(self, state: PipelineState) -> PipelineState:
        if not state.candidates:
            return state
        scores = await self._reranker.rerank(base_query(state), [c.text for c in state.candidates])
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = order[: state.config.rerank.top_n]
        ranked = tuple(state.candidates[i].model_copy(update={"score": scores[i]}) for i in top)
        return state.model_copy(update={"ranked": ranked, "reranker": self._reranker.name})
