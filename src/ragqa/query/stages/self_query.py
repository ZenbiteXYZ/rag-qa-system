"""Self-Query: the LLM extracts a semantic query and metadata filters, and every extracted
value is checked against the values that exist (ADR-0007, ADR-0008)."""

from collections.abc import Awaitable, Callable
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from ragqa.domain.conversation import ChatMessage
from ragqa.domain.pipeline import (
    Clarification,
    DeletedDocument,
    PipelineConfig,
    PipelineState,
    Substitution,
)
from ragqa.domain.retrieval import KnownValues, MetadataFilter
from ragqa.llm.types import LLMClient
from ragqa.query import prompts
from ragqa.query.filters import resolve_document, resolve_value
from ragqa.query.stages.common import model_for, with_call


class ExtractedFilters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    language: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    document_title: str | None = None


class SelfQueryOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    semantic_query: str = ""
    filters: ExtractedFilters = Field(default_factory=ExtractedFilters)


KnownValuesProvider = Callable[[], Awaitable[KnownValues]]


class SelfQueryStage:
    name = "self_query"
    reads = ("question", "rewritten_query")
    writes = (
        "semantic_query",
        "parsed_filters",
        "parsed_document_title",
        "resolved_filters",
        "resolved_document_title",
        "substitutions",
        "clarification",
        "deleted_document",
    )
    optional = True

    def __init__(self, llm: LLMClient, known_values: KnownValuesProvider) -> None:
        self._llm = llm
        self._known_values = known_values

    def enabled(self, config: PipelineConfig) -> bool:
        return config.self_query.enabled

    async def run(self, state: PipelineState) -> PipelineState:
        config = state.config.self_query
        question = state.rewritten_query or state.question
        output, completion = await self._llm.complete_structured(
            [
                ChatMessage(role="system", content=prompts.SELF_QUERY_SYSTEM),
                ChatMessage(role="user", content=question),
            ],
            SelfQueryOutput,
            model=model_for(config.model, state),
            max_tokens=400,
            reasoning_effort=state.config.reasoning_effort,
        )
        extracted = output.filters
        update: dict[str, object] = {
            "semantic_query": output.semantic_query.strip() or question,
            "parsed_filters": MetadataFilter(
                doc_type=extracted.doc_type,
                tags=tuple(extracted.tags),
                author=extracted.author,
                language=extracted.language,
                date_from=extracted.date_from,
                date_to=extracted.date_to,
            ),
            "parsed_document_title": extracted.document_title,
            "llm_calls": with_call(state, self.name, completion),
        }

        known = await self._known_values()
        substitutions: list[Substitution] = []
        warnings: list[str] = []
        resolved: dict[str, object] = {
            "date_from": extracted.date_from,
            "date_to": extracted.date_to,
        }

        def check(field: str, value: str, options: tuple[str, ...]) -> str | Clarification | None:
            result = resolve_value(value, options, config)
            substitutions.append(
                Substitution(
                    field=field, value=value, outcome=result.outcome, resolved=result.resolved
                )
            )
            if result.outcome == "substituted":
                warnings.append(f"{field} {value!r} was replaced by {result.resolved!r}")
            elif result.outcome == "unresolved":
                warnings.append(f"{field} {value!r} matches no known value and was dropped")
            if result.outcome == "asked":
                return Clarification(field=field, value=value, candidates=result.candidates)
            return result.resolved

        single = {
            "doc_type": (extracted.doc_type, known.doc_types),
            "author": (extracted.author, known.authors),
            "language": (extracted.language, known.languages),
        }
        for field, (value, options) in single.items():
            if value:
                outcome = check(field, value, options)
                if isinstance(outcome, Clarification):
                    return self._stop(state, update, substitutions, warnings, outcome)
                resolved[field] = outcome
        tags: list[str] = []
        for tag in extracted.tags:
            outcome = check("tags", tag, known.tags)
            if isinstance(outcome, Clarification):
                return self._stop(state, update, substitutions, warnings, outcome)
            if outcome:
                tags.append(outcome)
        resolved["tags"] = tuple(tags)

        if extracted.document_title:
            found = resolve_document(extracted.document_title, known.documents, config)
            substitutions.append(
                Substitution(
                    field="document",
                    value=extracted.document_title,
                    outcome=found.outcome,
                    resolved=found.matched_title,
                )
            )
            if found.outcome == "asked":
                clarification = Clarification(
                    field="document", value=extracted.document_title, candidates=found.candidates
                )
                return self._stop(state, update, substitutions, warnings, clarification)
            if found.deleted is not None:
                update["deleted_document"] = DeletedDocument(
                    document_id=found.deleted.id,
                    title=found.deleted.title,
                    deleted_at=found.deleted.deleted_at,  # type: ignore[arg-type]
                )
            elif found.live:
                resolved["document_ids"] = tuple(d.id for d in found.live)
                update["resolved_document_title"] = found.matched_title
                if found.outcome == "substituted":
                    warnings.append(
                        f"document {extracted.document_title!r} "
                        f"was replaced by {found.matched_title!r}"
                    )
            else:
                warnings.append(
                    f"document {extracted.document_title!r} matches nothing and was dropped"
                )

        update["resolved_filters"] = MetadataFilter.model_validate(resolved)
        update["substitutions"] = tuple(substitutions)
        update["warnings"] = (*state.warnings, *warnings)
        return state.model_copy(update=update)

    @staticmethod
    def _stop(
        state: PipelineState,
        update: dict[str, object],
        substitutions: list[Substitution],
        warnings: list[str],
        clarification: Clarification,
    ) -> PipelineState:
        update["clarification"] = clarification
        update["substitutions"] = tuple(substitutions)
        update["warnings"] = (*state.warnings, *warnings)
        return state.model_copy(update=update)
