import asyncio, json, logging
from datetime import datetime
from time import monotonic
from typing import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from azure_agent.api.schema import (
    JobCancelResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobStatusResponse,
)
from azure_agent.jobs.queue import (
    cancel_job,
    create_job,
    get_job,
    get_job_by_idempotency,
    read_events,
)
from azure_agent.session import SessionConflictError, SessionManager, SessionOwnershipError, SessionStatus

logger = logging.getLogger(__name__)

router = APIRouter()


def sse_pack(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def get_request_user_id(request: Request) -> str:
    user_id = str(request.headers.get("X-User-Id", "")).strip()
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "missing_user_identity",
                "message": "X-User-Id header is required",
            },
        )
    return user_id


def validate_job_owner(job: dict, request_user_id: str) -> None:
    if str(job.get("user_id", "")) != request_user_id:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "forbidden",
                "message": "job does not belong to the requesting user",
            },
        )


def build_job_create_response(request: Request, *, job_id: str, status: str) -> JobCreateResponse:
    return JobCreateResponse(
        job_id=UUID(job_id),
        status=status,
        status_url=str(request.url_for("get_job_endpoint", job_id=job_id)),
        events_url=str(request.url_for("stream_job_events", job_id=job_id)),
        cancel_url=str(request.url_for("cancel_job_endpoint", job_id=job_id)),
    )


@router.post(
    "/agent/api/jobs",
    response_model=JobCreateResponse,
    status_code=202,
    tags=["Jobs"],
)
async def create_job_endpoint(req: JobCreateRequest, request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    session_manager: SessionManager | None = getattr(request.app.state, "session_manager", None)
    runtime_config = getattr(request.app.state, "runtime_config", None)
    reservation_id: str | None = None
    request_user_id = get_request_user_id(request)
    thread_id = str(req.thread_id)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis stream client unavailable")
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager unavailable")
    if runtime_config is None:
        raise HTTPException(status_code=500, detail="Runtime config unavailable")

    try:
        if req.idempotency_key:
            existing_job = await get_job_by_idempotency(
                redis_client,
                user_id=request_user_id,
                idempotency_key=req.idempotency_key,
                request_stream_key="agent:requests",
                request_stream_maxlen=100_000,
                job_ttl_seconds=runtime_config.job.job_ttl_seconds,
            )
            if existing_job is not None:
                existing_job_id = str(existing_job.get("job_id", ""))
                if (
                    str(existing_job.get("thread_id", "")) != thread_id
                    or str(existing_job.get("user_id", "")) != request_user_id
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "idempotency_conflict",
                            "message": "idempotency_key is already bound to another job context",
                            "job_id": str(existing_job.get("job_id", "")),
                        },
                    )

                if str(existing_job.get("enqueue_state", "")) == "enqueue_failed":
                    await session_manager.clear_active_job(
                        thread_id=thread_id,
                        user_id=request_user_id,
                        expected_job_id=existing_job_id or None,
                        status=SessionStatus.idle,
                        last_job_id=existing_job_id or None,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "code": "job_enqueue_failed",
                            "message": "job enqueue failed; retry the request",
                            "job_id": existing_job_id,
                        },
                    )

                existing_status = str(existing_job.get("status", "queued"))
                active_job_id = (
                    str(existing_job.get("job_id", ""))
                    if existing_status in {"queued", "running"}
                    else None
                )
                await session_manager.upsert_session(
                    thread_id=thread_id,
                    user_id=request_user_id,
                    status=(
                        SessionStatus.queued
                        if existing_status == "queued"
                        else SessionStatus.running
                        if existing_status == "running"
                        else SessionStatus.idle
                    ),
                    last_job_id=str(existing_job.get("job_id", "")),
                    active_job_id=active_job_id,
                )
                return build_job_create_response(
                    request,
                    job_id=str(existing_job["job_id"]),
                    status=existing_status,
                )

        active_job_id = await session_manager.get_active_job(thread_id)
        if active_job_id:
            if session_manager.is_pending_job_ref(active_job_id):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "session_conflict",
                        "message": "session already has an in-flight job reservation",
                    },
                )

            active_job = await get_job(redis_client, active_job_id)
            if active_job is None or str(active_job.get("status", "")) in {
                "completed",
                "failed",
                "cancelled",
            }:
                await session_manager.clear_active_job(
                    thread_id=thread_id,
                    user_id=request_user_id,
                    expected_job_id=active_job_id,
                    status=SessionStatus.idle,
                    last_job_id=active_job_id,
                )
            else:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "session_conflict",
                        "message": "session already has an active job",
                        "active_job_id": active_job_id,
                    },
                )

        reservation_id = await session_manager.reserve_job(
            thread_id=thread_id,
            user_id=request_user_id,
        )
        job = await create_job(
            redis_client,
            thread_id=thread_id,
            user_id=request_user_id,
            user_query=req.user_query,
            idempotency_key=req.idempotency_key,
            job_ttl_seconds=runtime_config.job.job_ttl_seconds,
            idempotency_ttl_seconds=runtime_config.job.idempotency_ttl_seconds,
            request_stream_maxlen=100_000, # 100K
        )
        job_id = str(job["job_id"])
        status = str(job.get("status", "queued"))
        enqueue_state = str(job.get("enqueue_state", "enqueued"))
        if enqueue_state != "enqueued":
            await session_manager.clear_active_job(
                thread_id=thread_id,
                user_id=request_user_id,
                expected_job_id=reservation_id,
                status=SessionStatus.idle,
                last_job_id=job_id,
            )
            reservation_id = None
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "job_enqueue_failed",
                    "message": "job enqueue failed; retry the request",
                    "job_id": job_id,
                },
            )
        is_active = status in {"queued", "running"}

        if (
            not bool(job.get("created", True))
            and (
                str(job.get("thread_id", "")) != thread_id
                or str(job.get("user_id", "")) != request_user_id
            )
        ):
            await session_manager.clear_active_job(
                thread_id=thread_id,
                user_id=request_user_id,
                expected_job_id=reservation_id,
                status=SessionStatus.idle,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "idempotency_conflict",
                    "message": "idempotency_key resolved to another job context",
                    "job_id": job_id,
                },
            )

        if is_active:
            await session_manager.bind_job(
                thread_id=thread_id,
                user_id=request_user_id,
                reservation_id=reservation_id,
                job_id=job_id,
                status=SessionStatus.queued,
            )
            reservation_id = None
        else:
            await session_manager.clear_active_job(
                thread_id=thread_id,
                user_id=request_user_id,
                expected_job_id=reservation_id,
                status=SessionStatus.idle,
                last_job_id=job_id,
            )
            reservation_id = None
    except Exception as exc:
        if reservation_id is not None:
            try:
                await session_manager.clear_active_job(
                    thread_id=thread_id,
                    user_id=request_user_id,
                    expected_job_id=reservation_id,
                    status=SessionStatus.idle,
                )
            except Exception:
                pass
        if isinstance(exc, HTTPException):
            raise
        if isinstance(exc, SessionOwnershipError):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "session_ownership_error",
                    "message": str(exc),
                },
            ) from exc
        if isinstance(exc, SessionConflictError):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "session_conflict",
                    "message": str(exc),
                    "active_job_id": exc.active_job_id,
                },
            ) from exc
        logger.exception("[job.py] Failed to create job: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create job") from exc

    return build_job_create_response(
        request,
        job_id=job_id,
        status=status,
    )


@router.get(
    "/agent/api/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
)
async def get_job_endpoint(job_id: UUID, request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis stream client unavailable")
    request_user_id = get_request_user_id(request)
    job_id_str = str(job_id)
    job = await get_job(redis_client, job_id_str)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    validate_job_owner(job, request_user_id)

    try:
        return JobStatusResponse(
            job_id=UUID(str(job["job_id"])),
            status=str(job.get("status", "failed")),
            thread_id=UUID(str(job["thread_id"])),
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
        logger.exception("[job.py] Invalid job payload for %s: %s", job_id_str, exc)
        raise HTTPException(status_code=500, detail="Invalid job state") from exc


@router.get(
    "/agent/api/jobs/{job_id}/events",
    response_class=StreamingResponse,
    tags=["Jobs"],
)
async def stream_job_events(job_id: UUID, request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    runtime_config = getattr(request.app.state, "runtime_config", None)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis stream client unavailable")
    if runtime_config is None:
        raise HTTPException(status_code=500, detail="Runtime config unavailable")
    request_user_id = get_request_user_id(request)
    job_id_str = str(job_id)
    job = await get_job(redis_client, job_id_str)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    validate_job_owner(job, request_user_id)

    async def gen() -> AsyncGenerator[str, None]:
        last_id = request.headers.get("last-event-id") or "0-0"
        started_at = monotonic()
        max_connection_seconds = runtime_config.api.sse_max_connection_seconds
        try:
            while True:
                if await request.is_disconnected():
                    return
                if monotonic() - started_at >= max_connection_seconds:
                    logger.info(
                        "[job.py] SSE connection timeout for %s after %ss",
                        job_id_str,
                        max_connection_seconds,
                    )
                    return

                events = await read_events(
                    redis_client,
                    job_id_str,
                    last_id=last_id,
                )
                if not events:
                    current_job = await get_job(redis_client, job_id_str)
                    if current_job is None:
                        return
                    if str(current_job.get("status", "")) in {
                        "completed",
                        "failed",
                        "cancelled",
                    }:
                        drain_events = await read_events(
                            redis_client,
                            job_id_str,
                            last_id=last_id,
                            block_ms=0,
                        )
                        for evt in drain_events:
                            event_id = str(evt["id"])
                            payload = dict(evt["fields"])
                            payload["event_id"] = event_id
                            yield sse_pack(payload)
                            last_id = event_id

                            if payload.get("type") == "complete":
                                return
                        return
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
            logger.warning("[job.py] Failed to stream events for %s: %s", job_id_str, exc)
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
async def cancel_job_endpoint(job_id: UUID, request: Request):
    redis_client = getattr(request.app.state, "redis_stream_client", None)
    runtime_config = getattr(request.app.state, "runtime_config", None)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis stream client unavailable")
    if runtime_config is None:
        raise HTTPException(status_code=500, detail="Runtime config unavailable")
    request_user_id = get_request_user_id(request)
    job_id_str = str(job_id)
    job = await get_job(redis_client, job_id_str)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    validate_job_owner(job, request_user_id)

    ok = await cancel_job(
        redis_client,
        job_id_str,
        ttl_seconds=runtime_config.job.job_ttl_seconds,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")

    job = await get_job(redis_client, job_id_str)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobCancelResponse(
        job_id=job_id,
        cancel_requested=bool(job.get("cancel_requested", False)),
        status=str(job.get("status", "failed")),
    )
