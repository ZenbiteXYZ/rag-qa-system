"""Retrieval: dense, sparse or hybrid search over the index (ADR-0018)."""

from ragqa.domain.pipeline import PipelineConfig, PipelineState
from ragqa.domain.retrieval import RetrievalQuery
from ragqa.query.retriever import Retriever
from ragqa.query.stages.common import base_query, merge_filters


class RetrieveStage:
    name = "retrieve"
    reads = (
        "question",
        "rewritten_query",
        "semantic_query",
        "hypothetical_document",
        "user_filters",
        "resolved_filters",
        "version",
        "index",
    )
    writes = ("candidates",)
    optional = False

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def enabled(self, config: PipelineConfig) -> bool:
        return config.retrieve.enabled

    async def run(self, state: PipelineState) -> PipelineState:
        config = state.config.retrieve
        text = base_query(state)
        query = RetrievalQuery(
            mode=config.mode,
            # The dense branch may use the HyDE text, the sparse branch never does.
            dense_text=state.hypothetical_document or text,
            sparse_text=text,
            filters=merge_filters(state.user_filters, state.resolved_filters),
            version=state.version,
            branch_limit=config.branch_limit,
            top_k=config.top_k,
        )
        found = await self._retriever.search(state.index, query)
        return state.model_copy(update={"candidates": tuple(found)})
