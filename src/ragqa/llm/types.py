"""LLM client contract. The implementation talks to the LiteLLM proxy (ADR-0016).

`model` everywhere is a LiteLLM alias from `deploy/litellm/config.yaml`, never a provider
model id. The provider model that actually answered is reported in `Completion.resolved_model`.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ragqa.domain.conversation import ChatMessage, ToolCall

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The proxy rejected the request, for example an unknown model alias. Not worth retrying."""


class LLMUnavailable(LLMError):
    """The proxy or the provider failed or timed out (5xx, 429, connection errors)."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolSpec(_Model):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments


class Usage(_Model):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float | None = None  # if the proxy reports it


class Completion(_Model):
    text: str  # empty if the model only called tools
    tool_calls: tuple[ToolCall, ...] = ()
    model: str  # the alias that was requested
    resolved_model: str  # what the proxy says answered, for run records (ADR-0019)
    usage: Usage | None = None
    finish_reason: str | None = None  # "length" means the answer was cut by max_tokens


class TextDelta(_Model):
    text: str


# A stream yields deltas and ends with exactly one `Completion` holding the full text and
# the usage (with streaming, usage only comes if it is requested for the final chunk).
StreamEvent = TextDelta | Completion


class LLMClient(Protocol):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        reasoning_effort: str | None = None,
        seed: int | None = None,
    ) -> Completion: ...

    async def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        model: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
        seed: int | None = None,
    ) -> tuple[T, Completion]:
        """For Self-Query parsing and the judge: the reply validated against `schema`.

        Raise `LLMError` if the reply does not validate.
        """
        ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def models(self) -> list[str]:
        """Aliases the proxy serves. The API validates a user's model choice against this."""
        ...
