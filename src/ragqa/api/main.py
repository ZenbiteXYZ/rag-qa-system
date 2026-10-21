"""API app factory. Run with `uvicorn --factory ragqa.api.main:create_app`."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ragqa.api import conversations, documents, health, indexes, jobs, query
from ragqa.api.errors import register_error_handlers
from ragqa.api.readiness import build_checks
from ragqa.api.resources import Resources
from ragqa.observability.logging import configure_logging, get_logger
from ragqa.observability.middleware import RequestContextMiddleware
from ragqa.observability.tracing import instrument_fastapi, setup_tracing, shutdown_tracing
from ragqa.settings import Settings, get_settings

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # Blocking network calls to MLflow, so off the event loop.
    await asyncio.to_thread(setup_tracing, settings)
    resources = Resources.create(settings)
    app.state.resources = resources
    app.state.checks = build_checks(settings, resources)
    log.info("api_started", env=settings.env)
    try:
        yield
    finally:
        await resources.close()
        await asyncio.to_thread(shutdown_tracing)
        log.info("api_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # Configure logging before anything logs, including uvicorn's startup lines.
    configure_logging(
        service=settings.service_name,
        env=settings.env,
        level=settings.log_level,
        fmt=settings.log_format,
    )
    app = FastAPI(title="ragqa", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(health.router)
    for router in (
        documents.router,
        indexes.router,
        jobs.router,
        conversations.router,
        query.router,
    ):
        app.include_router(router)
    instrument_fastapi(app)
    return app
