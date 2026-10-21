import json
from collections.abc import Callable

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from ragqa.domain.conversation import ChatMessage, ToolCall
from ragqa.llm.client import ProxyLLMClient
from ragqa.llm.types import Completion, LLMError, LLMUnavailable, TextDelta, ToolSpec

Handler = Callable[[httpx.Request], httpx.Response]


def client_with(handler: Handler) -> ProxyLLMClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ProxyLLMClient(
        AsyncOpenAI(base_url="http://proxy/v1", api_key="k", http_client=http, max_retries=0)
    )


def chat_response(message: dict[str, object], finish: str = "stop") -> dict[str, object]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "anthropic/claude-haiku-5.5-20261007",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
    }


async def test_complete_sends_history_and_tools_in_wire_format() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json=chat_response({"role": "assistant", "content": "I. Petrov"})
        )

    call = ToolCall(id="c1", name="search_docs", arguments={"q": "ivan"})
    history = [
        ChatMessage(role="user", content="Who wrote it?"),
        ChatMessage(role="assistant", tool_calls=(call,)),
        ChatMessage(role="tool", content="[]", tool_call_id="c1"),
    ]
    tool = ToolSpec(name="search_docs", description="d", parameters={"type": "object"})
    result = await client_with(handler).complete(
        history, model="haiku-5.5", max_tokens=100, tools=[tool]
    )

    assert seen["model"] == "haiku-5.5"
    assert seen["max_completion_tokens"] == 100
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"q": "ivan"}'
    assert messages[1]["content"] is None
    assert messages[2] == {"role": "tool", "tool_call_id": "c1", "content": "[]"}
    assert seen["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert result.text == "I. Petrov"
    assert result.model == "haiku-5.5"
    assert result.resolved_model == "anthropic/claude-haiku-5.5-20261007"
    assert result.usage is not None and result.usage.prompt_tokens == 11


async def test_complete_parses_tool_calls() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c9",
                "type": "function",
                "function": {"name": "search_docs", "arguments": '{"q": "drain"}'},
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(message, finish="tool_calls"))

    result = await client_with(handler).complete(
        [ChatMessage(role="user", content="hi")], model="m", max_tokens=10
    )
    assert result.text == ""
    assert result.tool_calls == (ToolCall(id="c9", name="search_docs", arguments={"q": "drain"}),)
    assert result.finish_reason == "tool_calls"


async def test_bad_tool_arguments_are_an_llm_error() -> None:
    message = {
        "role": "assistant",
        "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "t", "arguments": "{oops"}}
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(message))

    with pytest.raises(LLMError, match="not valid JSON"):
        await client_with(handler).complete(
            [ChatMessage(role="user", content="hi")], model="m", max_tokens=10
        )


@pytest.mark.parametrize(
    ("status", "expected"), [(500, LLMUnavailable), (429, LLMUnavailable), (400, LLMError)]
)
async def test_http_errors_are_translated(status: int, expected: type[LLMError]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(expected) as info:
        await client_with(handler).complete(
            [ChatMessage(role="user", content="hi")], model="m", max_tokens=10
        )
    assert (type(info.value) is LLMUnavailable) == (expected is LLMUnavailable)


async def test_connection_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(LLMUnavailable):
        await client_with(handler).models()


class Verdict(BaseModel):
    correct: bool


async def test_structured_reply_is_validated() -> None:
    def reply(content: str) -> Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["response_format"]["json_schema"]["name"] == "Verdict"
            return httpx.Response(
                200, json=chat_response({"role": "assistant", "content": content})
            )

        return handler

    messages = [ChatMessage(role="user", content="judge")]
    verdict, completion = await client_with(reply('{"correct": true}')).complete_structured(
        messages, Verdict, model="m", max_tokens=10
    )
    assert verdict.correct is True
    assert isinstance(completion, Completion)
    with pytest.raises(LLMError, match="does not match Verdict"):
        await client_with(reply('{"correct": "maybe"}')).complete_structured(
            messages, Verdict, model="m", max_tokens=10
        )


async def test_stream_yields_deltas_then_completion_with_usage() -> None:
    def chunk(delta: dict[str, object], finish: str | None = None) -> str:
        data = {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "resolved-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(data)}\n\n"

    usage_chunk = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "resolved-model",
        "choices": [],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    }
    body = (
        chunk({"role": "assistant", "content": "Hel"})
        + chunk({"content": "lo"}, finish="stop")
        + f"data: {json.dumps(usage_chunk)}\n\n"
        + "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = [
        e
        async for e in client_with(handler).stream(
            [ChatMessage(role="user", content="hi")], model="m", max_tokens=10
        )
    ]
    assert events[:2] == [TextDelta(text="Hel"), TextDelta(text="lo")]
    final = events[2]
    assert isinstance(final, Completion)
    assert final.text == "Hello"
    assert final.resolved_model == "resolved-model"
    assert final.finish_reason == "stop"
    assert final.usage is not None and final.usage.completion_tokens == 2


async def test_models_lists_aliases_sorted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        data = [{"id": i, "object": "model", "created": 0, "owned_by": "x"} for i in ("b", "a")]
        return httpx.Response(200, json={"object": "list", "data": data})

    assert await client_with(handler).models() == ["a", "b"]


async def test_reasoning_effort_and_seed_are_sent_only_when_set() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"role": "assistant", "content": "ok"}))

    client = client_with(handler)
    messages = [ChatMessage(role="user", content="hi")]
    await client.complete(messages, model="m", max_tokens=5)
    await client.complete(messages, model="m", max_tokens=5, reasoning_effort="low", seed=7)

    assert "reasoning_effort" not in bodies[0] and "seed" not in bodies[0]
    assert bodies[1]["reasoning_effort"] == "low" and bodies[1]["seed"] == 7
