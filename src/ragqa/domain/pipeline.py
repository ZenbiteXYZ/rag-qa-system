"""Query pipeline state and config (ADR-0010).

Every stage gets the state and returns a new one with only its own fields changed
(`state.model_copy(update=...)`). The state is frozen, so a stage cannot change what
another stage wrote by accident, and each stage's output can be traced as a diff.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ragqa.domain.conversation import ChatMessage
from ragqa.domain.indexes import Index
from ragqa.domain.retrieval import MetadataFilter, RetrievedChunk, SearchMode


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Config: what a request (or an eval config file) switches on ---
# `model` is a LiteLLM alias. None means `PipelineConfig.default_model`.


class HistoryConfig(_Model):
    max_messages: int = Field(default=10, ge=0)  # the newest ones; 0 ignores history


class RewriteConfig(_Model):
    enabled: bool = False
    model: str | None = None


class SelfQueryConfig(_Model):
    enabled: bool = False
    model: str | None = None
    # TODO(author): fuzzy-match threshold and margin for "one clearly best candidate" (ADR-0008)


class HydeConfig(_Model):
    enabled: bool = False
    model: str | None = None


class RetrieveConfig(_Model):
    mode: SearchMode = "dense"  # the baseline is dense only (ADR-0019)
    branch_limit: int = Field(default=20, gt=0)
    top_k: int = Field(default=10, gt=0)


class RerankConfig(_Model):
    enabled: bool = False
    top_n: int = Field(default=5, gt=0)
    # TODO(author): reranker model and how it is served


class ContextConfig(_Model):
    max_tokens: int = Field(default=4000, gt=0)


class GenerateConfig(_Model):
    model: str | None = None
    max_tokens: int = Field(default=1024, gt=0)


class PipelineConfig(_Model):
    default_model: str
    history: HistoryConfig = HistoryConfig()
    rewrite: RewriteConfig = RewriteConfig()
    self_query: SelfQueryConfig = SelfQueryConfig()
    hyde: HydeConfig = HydeConfig()
    retrieve: RetrieveConfig = RetrieveConfig()
    rerank: RerankConfig = RerankConfig()
    context: ContextConfig = ContextConfig()
    generate: GenerateConfig = GenerateConfig()


# --- Outcomes that end the pipeline early (ARCHITECTURE section 5, `POST /query`) ---


class Clarification(_Model):
    """A filter value matched no known value and no single candidate was clearly best."""

    field: str
    value: str
    candidates: tuple[str, ...]


class DeletedDocument(_Model):
    document_id: uuid.UUID
    title: str
    deleted_at: datetime


class Citation(_Model):
    document_id: uuid.UUID
    version: int
    title: str
    section_path: tuple[str, ...]
    chunk_index: int
    quote: str


Outcome = Literal["answer", "needs_clarification", "document_deleted"]


class PipelineState(_Model):
    # Request (set by the API, read-only for stages)
    question: str = Field(min_length=1)
    index: Index
    user_filters: MetadataFilter = MetadataFilter()
    version: int | None = None
    # Earlier messages of the conversation, oldest first, tool calls and results included.
    # The current question is not in it. Rewrite reads it to make a standalone query, and
    # generate may include the newest `config.history.max_messages` of it in the prompt.
    history: tuple[ChatMessage, ...] = ()
    config: PipelineConfig
    trace_id: str | None = None

    # rewrite
    rewritten_query: str | None = None
    # self_query
    semantic_query: str | None = None  # the question without its filter cues
    parsed_filters: MetadataFilter | None = None  # as the LLM produced them
    resolved_filters: MetadataFilter | None = None  # after checking known values (ADR-0008)
    clarification: Clarification | None = None
    deleted_document: DeletedDocument | None = None
    # hyde
    hypothetical_document: str | None = None
    # retrieve
    candidates: tuple[RetrievedChunk, ...] = ()
    # rerank
    ranked: tuple[RetrievedChunk, ...] = ()
    # context
    context_chunks: tuple[RetrievedChunk, ...] = ()
    # generate
    answer: str | None = None
    citations: tuple[Citation, ...] = ()

    # Any stage: notes for the trace and the response, like a filter value substitution
    warnings: tuple[str, ...] = ()

    @property
    def outcome(self) -> Outcome | None:
        """Set once the pipeline has a result; the runner stops at the first non-None."""
        if self.deleted_document is not None:
            return "document_deleted"
        if self.clarification is not None:
            return "needs_clarification"
        if self.answer is not None:
            return "answer"
        return None
