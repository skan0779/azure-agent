import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from azure_agent.api.schema import HealthResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/agent/api/health", response_model=HealthResponse, tags=["Health"])
async def health(request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    session_manager = getattr(request.app.state, "session_manager", None)
    runtime_config = getattr(request.app.state, "runtime_config", None)

    checks = {
        "runtime_config": runtime_config is not None,
        "session_manager": session_manager is not None,
        "redis_client": redis_client is not None,
        "redis_ping": False,
    }

    if redis_client is not None:
        try:
            checks["redis_ping"] = bool(await redis_client.ping())
        except Exception as exc:
            logger.warning("[health.py] Redis readiness ping failed: %s", exc)

    ready = all(checks.values())
    status_code = 200 if ready else 503              

    return JSONResponse(
        status_code=status_code,
        content=HealthResponse(
            status=ready,
            checks=checks,
        ).model_dump(),
    )
