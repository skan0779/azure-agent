from fastapi import APIRouter
from fastapi.responses import JSONResponse

from schemas.api import PingResponse


router = APIRouter()


@router.get("/agent/api/ping", response_model=PingResponse, tags=["Ping"])
async def ping():
    return JSONResponse({"ping": True})
