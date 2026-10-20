from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from ragqa.api.main import create_app
from ragqa.api.readiness import Check
from ragqa.settings import Settings
from tests.conftest import LogLines, log_buffer


class Item(BaseModel):
    name: str
    password: str


def _add_test_routes(app: FastAPI) -> None:
    @app.get("/boom/{item_id}")
    async def boom(item_id: int) -> None:
        raise RuntimeError("postgresql://u:leaked-pass@db/x is down")

    @app.post("/items")
    async def items(item: Item) -> dict[str, str]:
        return {"name": item.name}


async def _ok() -> None:
    return None


async def _fail() -> None:
    raise ConnectionError("refused")


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    application = create_app(settings)
    _add_test_routes(application)
    return application


@pytest.fixture
def app_logs(app: FastAPI) -> Iterator[LogLines]:
    """Log buffer for API tests. Set up after `app`, because create_app configures logging."""
    yield from log_buffer()


@pytest.fixture
def client(app: FastAPI, app_logs: LogLines) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_id_is_generated_and_returned(client: TestClient) -> None:
    response = client.get("/health")

    assert len(response.headers["X-Request-ID"]) == 32


def test_valid_incoming_request_id_is_kept(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})

    assert response.headers["X-Request-ID"] == "abc-123"


def test_unsafe_incoming_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "x" * 200})

    assert response.headers["X-Request-ID"] != "x" * 200


def test_ready_when_required_checks_pass(app: FastAPI, client: TestClient) -> None:
    app.state.checks = [Check("postgres", True, _ok), Check("embedder", False, _fail)]

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["postgres"]["status"] == "ok"
    assert body["checks"]["embedder"] == {
        "status": "fail",
        "required": False,
        "latency_ms": body["checks"]["embedder"]["latency_ms"],
        "error": "ConnectionError: refused",
    }


def test_not_ready_when_a_required_check_fails(app: FastAPI, client: TestClient) -> None:
    app.state.checks = [Check("postgres", True, _ok), Check("rabbitmq", True, _fail)]

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_metrics_endpoint_exposes_http_metrics(client: TestClient) -> None:
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert 'ragqa_http_requests_total{method="GET",route="/health",status="200"}' in response.text


def test_unhandled_error_returns_json_500_and_logs(client: TestClient, app_logs: LogLines) -> None:
    response = client.get("/boom/1", headers={"X-Request-ID": "req-boom"})

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-boom"
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Internal server error"},
        "request_id": "req-boom",
    }
    lines = [line for line in app_logs() if line.get("request_id") == "req-boom"]
    error = next(line for line in lines if line["event"] == "unhandled_error")
    access = next(line for line in lines if line["event"] == "http_request")
    assert "RuntimeError" in str(error["exception"])
    assert "leaked-pass" not in str(error["exception"])
    assert access["status"] == 500
    assert access["route"] == "/boom/{item_id}"


def test_access_log_line(client: TestClient, app_logs: LogLines) -> None:
    client.get("/health?token=abc", headers={"X-Request-ID": "req-log"})

    (line,) = [x for x in app_logs() if x["event"] == "http_request"]
    assert line["request_id"] == "req-log"
    assert line["level"] == "debug"  # probes are logged at debug
    assert line["path"] == "/health"  # no query string
    assert line["status"] == 200
    assert isinstance(line["duration_ms"], float)


def test_not_found_is_json_with_unmatched_route_label(client: TestClient) -> None:
    response = client.get("/no/such/path", headers={"X-Request-ID": "req-404"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["request_id"] == "req-404"
    metrics_text = client.get("/metrics").text
    assert 'route="__unmatched__",status="404"' in metrics_text
    assert "/no/such/path" not in metrics_text


def test_validation_error_does_not_echo_input(client: TestClient) -> None:
    response = client.post("/items", json={"name": 1, "password": None})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert all(set(d) == {"loc", "msg", "type"} for d in body["error"]["details"])
