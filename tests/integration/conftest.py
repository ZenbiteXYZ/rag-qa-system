"""Fixtures for tests that need a real Postgres. The tests reset the schema, so the database
name must contain "test"."""

import os
from collections.abc import AsyncIterator, Iterator

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.engine import make_url

from ragqa.storage.postgres.migrate import alembic_config
from ragqa.storage.postgres.registry import Registry, create_session_factory

POSTGRES_DSN = os.environ.get("RAGQA_TEST_POSTGRES_DSN")
QDRANT_URL = os.environ.get("RAGQA_TEST_QDRANT_URL")


@pytest.fixture(scope="module")
def migrated() -> Iterator[str]:
    assert POSTGRES_DSN is not None
    if "test" not in (make_url(POSTGRES_DSN).database or ""):
        pytest.fail("refusing to reset a database whose name does not contain 'test'")
    config = alembic_config(POSTGRES_DSN)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    yield POSTGRES_DSN


@pytest.fixture
async def registry(migrated: str) -> AsyncIterator[Registry]:
    engine, sessions = create_session_factory(migrated)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE messages, conversations, jobs, version_index_builds, versions, "
                "documents, indexes RESTART IDENTITY CASCADE"
            )
        )
    yield Registry(sessions)
    await engine.dispose()
