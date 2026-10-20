"""ASGI middleware: request id, access log, HTTP metrics, last-resort error handler.

Pure ASGI rather than `BaseHTTPMiddleware`, so streamed (SSE) responses pass through
unbuffered and contextvars bound here are visible inside the endpoint.
"""

import json
import time

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ragqa.observability import context
from ragqa.observability.logging import get_logger
from ragqa.observability.metrics import UNMATCHED_ROUTE, record_http

REQUEST_ID_HEADER = "X-Request-ID"

# Probes and scrapes are logged at debug level so they do not flood the logs.
_QUIET_PATHS = frozenset({"/health", "/ready", "/metrics"})

log = get_logger("ragqa.access")


def route_template(scope: Scope) -> str:
    """The matched route's path template (`/documents/{id}`), never the raw path."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else UNMATCHED_ROUTE


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = context.valid_request_id(incoming) or context.new_request_id()
        context.bind_request(request_id)

        status = 500
        response_started = False
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status = message["status"]
                response_started = True
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            log.exception("unhandled_error", path=scope["path"])
            span = trace.get_current_span()
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            if response_started:
                raise
            status = 500
            await _send_error(send, request_id)
        finally:
            duration = time.perf_counter() - start
            method = scope["method"]
            record_http(method, route_template(scope), status, duration)
            log_method = log.debug if scope["path"] in _QUIET_PATHS else log.info
            log_method(
                "http_request",
                method=method,
                path=scope["path"],  # without the query string, which may carry secrets
                route=route_template(scope),
                status=status,
                duration_ms=round(duration * 1000, 1),
            )
            context.clear()


async def _send_error(send: Send, request_id: str) -> None:
    body = json.dumps(
        {
            "error": {"code": "internal_error", "message": "Internal server error"},
            "request_id": request_id,
        }
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (REQUEST_ID_HEADER.lower().encode(), request_id.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
