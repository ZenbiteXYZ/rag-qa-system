"""JSON error responses with the request id. Unhandled errors are caught by the middleware."""

from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from structlog.contextvars import get_contextvars

from ragqa.embeddings.tei import EmbedderError, EmbedderUnavailable
from ragqa.ingestion.errors import UnsupportedFormatError
from ragqa.llm.types import LLMError, LLMUnavailable
from ragqa.query.service import QueryRequestError
from ragqa.storage.postgres.registry import DuplicateIndexError, NotFoundError


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    body["request_id"] = get_contextvars().get("request_id")
    return body


def _code_for(status: int) -> str:
    try:
        return HTTPStatus(status).phrase.lower().replace(" ", "_").replace("-", "_")
    except ValueError:
        return "error"


async def http_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return JSONResponse(
        error_body(_code_for(exc.status_code), str(exc.detail)),
        status_code=exc.status_code,
        headers=exc.headers,
    )


async def validation_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Leave out `input`: it echoes the request body, which may contain secrets.
    details = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(
        error_body("validation_error", "Request validation failed", details), status_code=422
    )


def _handler(status: int, code: str) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(error_body(code, str(exc)), status_code=status)

    return handle


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(NotFoundError, _handler(404, "not_found"))
    app.add_exception_handler(QueryRequestError, _handler(422, "invalid_query"))
    app.add_exception_handler(LLMUnavailable, _handler(503, "llm_unavailable"))
    app.add_exception_handler(LLMError, _handler(502, "llm_error"))
    app.add_exception_handler(DuplicateIndexError, _handler(409, "index_exists"))
    app.add_exception_handler(UnsupportedFormatError, _handler(415, "unsupported_format"))
    app.add_exception_handler(EmbedderError, _handler(502, "embedder_error"))
    app.add_exception_handler(EmbedderUnavailable, _handler(503, "embedder_unavailable"))
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
