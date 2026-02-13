import asyncio, logging
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from jobs.job_queue import cancel_job, create_job, get_job, read_events

from schemas.api import JobCancelResponse, JobCreateRequest, JobCreateResponse, JobStatusResponse

from utils.sse import sse_pack

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/agent/api/jobs",
    response_model=JobCreateResponse,
    status_code=202,
    tags=["Jobs"],
)
async def create_job_endpoint(req: JobCreateRequest, request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis stream client unavailable")

    try:
        job = await create_job(
            redis_client,
            thread_id=req.thread_id,
            user_id=req.user_id,
            user_query=req.user_query,
            idempotency_key=req.idempotency_key,
        )
    except Exception as exc:
        logger.exception("[job.py] Failed to create job: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create job") from exc

    job_id = str(job["job_id"])
    status = str(job.get("status", "queued"))
    return JobCreateResponse(
        job_id=job_id,
        status=status,
        status_url=str(request.url_for("get_job_endpoint", job_id=job_id)),
        events_url=str(request.url_for("stream_job_events", job_id=job_id)),
        cancel_url=str(request.url_for("cancel_job_endpoint", job_id=job_id)),
    )


@router.get(
    "/agent/api/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
)
async def get_job_endpoint(job_id: str, request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis stream client unavailable")
    job = await get_job(redis_client, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        return JobStatusResponse(
            job_id=str(job["job_id"]),
            status=str(job.get("status", "failed")),
            thread_id=str(job["thread_id"]),
            user_id=str(job["user_id"]),
            created_at=datetime.fromisoformat(str(job["created_at"]).replace("Z", "+00:00")),
            started_at=(
                datetime.fromisoformat(str(job["started_at"]).replace("Z", "+00:00"))
                if job.get("started_at")
                else None
            ),
            finished_at=(
                datetime.fromisoformat(str(job["finished_at"]).replace("Z", "+00:00"))
                if job.get("finished_at")
                else None
            ),
            error=str(job["error"]) if job.get("error") is not None else None,
            metadata=job.get("metadata") if isinstance(job.get("metadata"), dict) else None,
        )
    except Exception as exc:
        logger.exception("[job.py] Invalid job payload for %s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail="Invalid job state") from exc


@router.get(
    "/agent/api/jobs/{job_id}/events",
    response_class=StreamingResponse,
    tags=["Jobs"],
)
async def stream_job_events(job_id: str, request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis stream client unavailable")

    async def gen() -> AsyncGenerator[str, None]:
        last_id = request.headers.get("last-event-id") or "0-0"
        try:
            while True:
                if await request.is_disconnected():
                    return

                events = await read_events(
                    redis_client,
                    job_id,
                    last_id=last_id,
                )
                if not events:
                    # Keep connection alive for proxies/LB
                    yield ": ping\n\n"
                    continue

                for evt in events:
                    event_id = str(evt["id"])
                    payload = dict(evt["fields"])
                    payload["event_id"] = event_id
                    yield sse_pack(payload)
                    last_id = event_id

                    if payload.get("type") == "complete":
                        return
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("[job.py] Failed to stream events for %s: %s", job_id, exc)
            yield sse_pack({"type": "error", "content": "Event stream error"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/agent/api/jobs/{job_id}/cancel",
    response_model=JobCancelResponse,
    tags=["Jobs"],
)
async def cancel_job_endpoint(job_id: str, request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis stream client unavailable")

    ok = await cancel_job(redis_client, job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")

    job = await get_job(redis_client, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobCancelResponse(
        job_id=job_id,
        cancel_requested=bool(job.get("cancel_requested", False)),
        status=str(job.get("status", "failed")),
    )
