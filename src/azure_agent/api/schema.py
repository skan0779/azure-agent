from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PingResponse(BaseModel):
    status: bool = Field(..., description="liveness status")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": True,
            }
        }
    )


class HealthResponse(BaseModel):
    status: bool = Field(..., description="readiness status")
    checks: dict[str, bool] = Field(..., description="dependency check results")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": True,
                "checks": {
                    "runtime_config": True,
                    "session_manager": True,
                    "redis_client": True,
                    "redis_ping": True,
                },
            }
        }
    )


class ErrorDetail(BaseModel):
    code: str | None = Field(None, description="machine-readable error code")
    message: str = Field(..., description="human-readable error message")
    job_id: UUID | None = Field(None, description="related job ID when available")
    active_job_id: UUID | None = Field(None, description="active job ID when available")


class ErrorResponse(BaseModel):
    detail: str | ErrorDetail = Field(..., description="error payload")
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detail": {
                        "code": "missing_user_identity",
                        "message": "X-User-Id header is required",
                    }
                },
                {
                    "detail": "Job not found",
                },
            ]
        }
    )


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobCreateRequest(BaseModel):
    thread_id: UUID = Field(..., description="thread/session ID")
    user_query: str = Field(..., description="user query")
    idempotency_key: str | None = Field(None, description="dedupe key (optional)")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thread_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
                "user_query": "Hello?",
                "idempotency_key": "req-20260211-0001",
            }
        }
    )


class JobCreateResponse(BaseModel):
    job_id: UUID = Field(..., description="job ID")
    status: JobStatus = Field(JobStatus.queued, description="job status")
    status_url: str = Field(..., description="GET job status endpoint")
    events_url: str = Field(..., description="SSE events endpoint")
    cancel_url: str = Field(..., description="cancel endpoint")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
                "status": "queued",
                "status_url": "https://api.example.com/agent/api/jobs/44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
                "events_url": "https://api.example.com/agent/api/jobs/44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6/events",
                "cancel_url": "https://api.example.com/agent/api/jobs/44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6/cancel",
            }
        }
    )


class JobStatusResponse(BaseModel):
    job_id: UUID = Field(..., description="job ID")
    status: JobStatus = Field(..., description="job status")
    thread_id: UUID = Field(..., description="thread/session ID")
    user_id: str = Field(..., description="user ID")
    created_at: datetime = Field(..., description="timezone-aware UTC timestamp")
    started_at: datetime | None = Field(None, description="timezone-aware UTC timestamp")
    finished_at: datetime | None = Field(None, description="timezone-aware UTC timestamp")
    error: str | None = Field(None, description="error message if failed")
    metadata: dict[str, Any] | None = Field(None, description="optional metadata")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
                "status": "running",
                "thread_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
                "user_id": "user-123",
                "created_at": "2026-04-02T10:00:00+00:00",
                "started_at": "2026-04-02T10:00:01+00:00",
                "finished_at": None,
                "error": None,
                "metadata": {
                    "source": "api",
                },
            }
        }
    )


class JobCancelResponse(BaseModel):
    job_id: UUID = Field(..., description="job ID")
    cancel_requested: bool = Field(..., description="cancel flag written")
    status: JobStatus = Field(..., description="current job status")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
                "cancel_requested": True,
                "status": "running",
            }
        }
    )
