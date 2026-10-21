"""Registry tables (ARCHITECTURE section 2), one module per area.

Importing this package registers every table on `Base.metadata`, which Alembic reads.
The migrations in `../migrations/versions` must match; an integration test runs `alembic check`.
"""

from ragqa.storage.postgres.models.base import Base
from ragqa.storage.postgres.models.conversations import Conversation, Message
from ragqa.storage.postgres.models.documents import Document, Version
from ragqa.storage.postgres.models.indexes import IndexRecord, VersionIndexBuild
from ragqa.storage.postgres.models.jobs import Job

__all__ = [
    "Base",
    "Conversation",
    "Document",
    "IndexRecord",
    "Job",
    "Message",
    "Version",
    "VersionIndexBuild",
]
