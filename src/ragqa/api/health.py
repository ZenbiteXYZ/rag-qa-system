"""Operational endpoints: liveness, readiness, Prometheus metrics."""

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ragqa.api.readiness import ReadyResponse, run_checks

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up. Does not touch dependencies."""
    return {"status": "ok"}


@router.get("/ready", responses={503: {"model": ReadyResponse}})
async def ready(request: Request, response: Response) -> ReadyResponse:
    """Readiness: required dependencies answer. 503 if any of them fails."""
    result = await run_checks(request.app.state.checks)
    if result.status != "ready":
        response.status_code = 503
    return result


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
