"""Chat messages, including tool calls and their results (story C4).

One shape for the pipeline's history, the LLM client and the stored conversation, so a
turn that used a tool (for example a `search_docs` lookup, ADR-0008) replays to the model
exactly as it happened. The fixed pipeline (ADR-0010) makes no tool calls yet, but a later
agent experiment will, and its history must already fit.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolCall(_Model):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(_Model):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant messages only
    tool_call_id: str | None = None  # tool messages only: the call this result answers

    @model_validator(mode="after")
    def _check_role_fields(self) -> Self:
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages can have tool_calls")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("tool messages need tool_call_id, and only they have it")
        if self.role != "assistant" and not self.content:
            raise ValueError(f"{self.role} message needs content")
        if self.role == "assistant" and not (self.content or self.tool_calls):
            raise ValueError("assistant message needs content or tool_calls")
        return self


class NewMessage(_Model):
    """A message to append to a conversation, with what the API knows about it."""

    message: ChatMessage
    trace_id: str | None = None  # of the request that produced an assistant message
    # Sources of an assistant answer, as `Citation` dicts. Shown to the client, never
    # sent back to the model.
    citations: tuple[dict[str, Any], ...] = ()


class StoredMessage(NewMessage):
    id: uuid.UUID
    conversation_id: uuid.UUID
    position: int = Field(ge=0)  # order inside the conversation, from 0
    created_at: datetime


def text_turns(history: Sequence[ChatMessage]) -> tuple[ChatMessage, ...]:
    """The conversation as plain text: tool results and tool calls are dropped.

    For stages that only need what was said (query rewriting), so tool arguments and
    results never reach their prompts. An assistant message that had both text and tool
    calls keeps its text.
    """
    return tuple(
        m.model_copy(update={"tool_calls": ()}) if m.role == "assistant" else m
        for m in history
        if m.role in ("user", "assistant") and (m.content or m.role == "user")
    )
