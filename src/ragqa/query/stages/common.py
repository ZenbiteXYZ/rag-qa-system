"""Helpers shared by the stages."""

from ragqa.domain.conversation import ChatMessage, text_turns
from ragqa.domain.pipeline import LlmCallRecord, PipelineState
from ragqa.domain.retrieval import MetadataFilter
from ragqa.llm.types import Completion


def model_for(configured: str | None, state: PipelineState) -> str:
    return configured or state.config.default_model


def with_call(
    state: PipelineState, stage: str, completion: Completion
) -> tuple[LlmCallRecord, ...]:
    usage = completion.usage
    record = LlmCallRecord(
        stage=stage,
        model=completion.model,
        resolved_model=completion.resolved_model,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
    )
    return (*state.llm_calls, record)


def base_query(state: PipelineState) -> str:
    """The text retrieval and reranking search with: the Self-Query semantic query, else the
    rewritten query, else the question (ARCHITECTURE section 4)."""
    return state.semantic_query or state.rewritten_query or state.question


def recent_text_turns(state: PipelineState) -> list[ChatMessage]:
    """The newest `history.max_messages` messages without tool calls and tool results."""
    limit = state.config.history.max_messages
    turns = text_turns(state.history)
    return list(turns[-limit:]) if limit else []


def merge_filters(user: MetadataFilter, automatic: MetadataFilter | None) -> MetadataFilter:
    """Filters the user set win over the ones Self-Query extracted, field by field."""
    if automatic is None:
        return user
    values = automatic.model_dump()
    values.update({k: v for k, v in user.model_dump().items() if v not in (None, ())})
    return MetadataFilter.model_validate(values)


def effective_filters(state: PipelineState) -> MetadataFilter:
    """What retrieval actually filtered on: user filters win over Self-Query's, field by field."""
    return merge_filters(state.user_filters, state.resolved_filters)
