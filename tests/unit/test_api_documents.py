"""Upload validation that happens before any database access."""

from collections.abc import Iterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragqa.api.main import create_app
from ragqa.settings import Settings
from tests.conftest import LogLines, log_buffer


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings.model_copy(update={"max_upload_bytes": 100}))


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # Set up after `app`, because create_app configures logging.
    logs: Iterator[LogLines] = log_buffer()
    next(logs)
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, name: str, content: bytes, **form: str) -> httpx.Response:
    return client.post("/documents", files={"file": (name, content)}, data=form)


def test_unsupported_extension_is_415(client: TestClient) -> None:
    response = _upload(client, "slides.pptx", b"x")

    assert response.status_code == 415
    error = response.json()["error"]
    assert error["code"] == "unsupported_format"
    assert ".md" in error["message"]


def test_too_large_is_413(client: TestClient) -> None:
    assert _upload(client, "big.md", b"x" * 101).status_code == 413


def test_empty_file_is_422(client: TestClient) -> None:
    assert _upload(client, "empty.md", b"").status_code == 422


def test_invalid_metadata_is_422_and_names_the_field(client: TestClient) -> None:
    response = _upload(client, "a.md", b"# A", metadata='{"doc_date": "yesterday"}')

    assert response.status_code == 422
    assert "doc_date" in response.json()["error"]["message"]


def test_unknown_metadata_field_is_422(client: TestClient) -> None:
    assert _upload(client, "a.md", b"# A", metadata='{"colour": "red"}').status_code == 422
