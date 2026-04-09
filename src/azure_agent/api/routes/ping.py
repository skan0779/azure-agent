from fastapi import APIRouter
from fastapi.responses import JSONResponse

from azure_agent.api.schema import PingResponse


router = APIRouter()


@router.get(
    "/agent/api/ping",
    response_model=PingResponse,
    tags=["Ping"],
    summary="Liveness check",
    description="Returns a simple boolean status to confirm that the API process is reachable.",
    response_description="Liveness result.",
)
async def ping():
    return JSONResponse({"status": True})
