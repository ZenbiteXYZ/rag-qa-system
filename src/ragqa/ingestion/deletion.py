"""Removing a document's data after it was tombstoned (Flow 7, stories A6).

Order: vectors, then raw files, then the version rows. Every step can be repeated, so a
crash anywhere is finished by the maintenance loop.
"""

import uuid

from ragqa.observability.logging import get_logger
from ragqa.storage.files.store import FileStore
from ragqa.storage.postgres.registry import Registry
from ragqa.storage.qdrant.store import ChunkStore

log = get_logger(__name__)


async def purge_document(
    registry: Registry, files: FileStore, chunk_store: ChunkStore, document_id: uuid.UUID
) -> None:
    for index in await registry.list_indexes():
        await chunk_store.delete_document(index, document_id)
    await files.delete_document(document_id)
    await registry.purge_versions(document_id)
    log.info("document_purged", document_id=str(document_id))
