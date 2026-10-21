"""Stored conversations (story C4)."""

import uuid

from fastapi import APIRouter

from ragqa.api.deps import ResourcesDep
from ragqa.domain.conversation import StoredMessage

router = APIRouter(tags=["conversations"])


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: uuid.UUID, resources: ResourcesDep) -> list[StoredMessage]:
    return await resources.backends.registry.get_messages(conversation_id)
