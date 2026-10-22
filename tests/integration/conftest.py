"""Fixtures for tests that need a real Postgres. The tests reset the schema, so the database
name must contain "test"."""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.engine import make_url

from ragqa.api.main import create_app
from ragqa.api.resources import Resources
from ragqa.settings import Settings
from ragqa.storage.postgres.migrate import alembic_config
from ragqa.storage.postgres.registry import Registry, create_session_factory
from tests.integration.fakes import FakeEmbedder, make_chunkers, word_counter_factory  # noqa: E402

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


@pytest.fixture
async def app(registry: Registry, tmp_path: Path) -> AsyncIterator[FastAPI]:
    assert POSTGRES_DSN is not None and QDRANT_URL is not None
    settings = Settings(
        postgres_dsn=POSTGRES_DSN,
        rabbitmq_url="amqp://nobody:nothing@localhost:1/",
        qdrant_url=QDRANT_URL,
        files_dir=tmp_path,
        env="test",
        service_name="ragqa-test",
        allow_query_debug=True,
    )
    application = create_app(settings)
    resources = Resources.create(settings)
    resources.backends.embedder = FakeEmbedder()
    resources.backends.chunkers = make_chunkers()
    resources.backends.token_counter = word_counter_factory()
    application.state.resources = resources
    yield application
    for index in await resources.backends.registry.list_indexes():
        await resources.backends.qdrant.delete_collection(index.collection)
    await resources.close()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
