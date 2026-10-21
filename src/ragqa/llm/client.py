"""LLM client for the LiteLLM proxy over its OpenAI-compatible API (ADR-0016).

The app holds no provider keys: the proxy resolves a model alias and talks to the provider.
Every call goes through `llm_call`, so latency, tokens and failures are recorded.
"""

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import ValidationError

from ragqa.domain.conversation import ChatMessage, ToolCall
from ragqa.llm.types import (
    Completion,
    LLMError,
    LLMUnavailable,
    StreamEvent,
    T,
    TextDelta,
    ToolSpec,
    Usage,
)
from ragqa.observability.tracing import LlmCall, llm_call
from ragqa.settings import Settings


def message_to_openai(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    payload: dict[str, Any] = {"role": message.role, "content": message.content or None}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                # The wire format carries the arguments as a JSON string.
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return payload


def _options(reasoning_effort: str | None, seed: int | None) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    if seed is not None:
        options["seed"] = seed
    return options


def _wire_messages(messages: Sequence[ChatMessage]) -> list[Any]:
    # `Any` because the SDK's message types are TypedDict unions that plain dicts do not match.
    return [message_to_openai(m) for m in messages]


def tool_to_openai(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_tool_call(call_id: str, name: str, arguments: str) -> ToolCall:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        raise LLMError(f"tool call {name!r} has arguments that are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"tool call {name!r} has arguments that are not a JSON object")
    return ToolCall(id=call_id, name=name, arguments=parsed)


def _usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    return Usage(prompt_tokens=raw.prompt_tokens, completion_tokens=raw.completion_tokens)


def _record_usage(call: LlmCall, usage: Usage | None) -> None:
    if usage is not None:
        call.prompt_tokens = usage.prompt_tokens
        call.completion_tokens = usage.completion_tokens


@contextmanager
def _translate_errors() -> Iterator[None]:
    """openai exceptions to ours: retryable failures become `LLMUnavailable`."""
    try:
        yield
    except openai.APIStatusError as exc:
        if exc.status_code >= 500 or exc.status_code == 429:
            raise LLMUnavailable(f"the LLM proxy returned {exc.status_code}") from exc
        raise LLMError(
            f"the LLM proxy rejected the request: {exc.status_code} {exc.message}"
        ) from exc
    except openai.APIConnectionError as exc:  # includes timeouts
        raise LLMUnavailable(f"the LLM proxy is not reachable: {type(exc).__name__}") from exc


class ProxyLLMClient:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProxyLLMClient":
        key = settings.litellm_api_key
        return cls(
            AsyncOpenAI(
                base_url=settings.litellm_url.rstrip("/") + "/v1",
                # The client refuses to start without a key. A proxy without a master key
                # ignores it.
                api_key=key.get_secret_value() if key else "unset",
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
        )

    async def close(self) -> None:
        await self._client.close()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        reasoning_effort: str | None = None,
        seed: int | None = None,
    ) -> Completion:
        extra = _options(reasoning_effort, seed)
        if tools:
            extra["tools"] = [tool_to_openai(t) for t in tools]
        return await self._complete(messages, model, max_tokens, extra)

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
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        extra = _options(reasoning_effort, seed) | {"response_format": response_format}
        completion = await self._complete(messages, model, max_tokens, extra)
        try:
            return schema.model_validate_json(completion.text), completion
        except ValidationError as exc:
            raise LLMError(f"the reply does not match {schema.__name__}: {exc}") from exc

    async def _complete(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        max_tokens: int,
        extra: dict[str, Any],
    ) -> Completion:
        with llm_call(model) as call, _translate_errors():
            response = await self._client.chat.completions.create(
                model=model,
                messages=_wire_messages(messages),
                max_completion_tokens=max_tokens,
                **extra,
            )
            if not response.choices:
                raise LLMError("the LLM proxy returned no choices")
            choice = response.choices[0]
            tool_calls = tuple(
                _parse_tool_call(c.id, c.function.name, c.function.arguments)
                for c in choice.message.tool_calls or ()
                if c.type == "function"
            )
            completion = Completion(
                text=choice.message.content or "",
                tool_calls=tool_calls,
                model=model,
                resolved_model=response.model,
                usage=_usage(response.usage),
                finish_reason=choice.finish_reason,
            )
            _record_usage(call, completion.usage)
            return completion

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Text deltas, then one `Completion` with the full text and the usage."""
        parts: list[str] = []
        resolved = model
        finish_reason: str | None = None
        usage: Usage | None = None
        with llm_call(model) as call, _translate_errors():
            chunks = await self._client.chat.completions.create(
                model=model,
                messages=_wire_messages(messages),
                max_completion_tokens=max_tokens,
                stream=True,
                **_options(reasoning_effort, None),
                # Without this the last chunk carries no token counts.
                stream_options={"include_usage": True},
            )
            async for chunk in chunks:
                resolved = chunk.model or resolved
                usage = _usage(chunk.usage) or usage
                for choice in chunk.choices:
                    if choice.delta.content:
                        parts.append(choice.delta.content)
                        yield TextDelta(text=choice.delta.content)
                    finish_reason = choice.finish_reason or finish_reason
            _record_usage(call, usage)
        yield Completion(
            text="".join(parts),
            model=model,
            resolved_model=resolved,
            usage=usage,
            finish_reason=finish_reason,
        )

    async def models(self) -> list[str]:
        """Aliases the proxy serves. The API validates a user's model choice against this."""
        with _translate_errors():
            page = await self._client.models.list()
        return sorted(m.id for m in page.data)
