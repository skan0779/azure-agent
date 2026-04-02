from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PingResponse(BaseModel):
    status: bool = Field(..., description="liveness status")


class HealthResponse(BaseModel):
    status: bool = Field(..., description="readiness status")
    checks: dict[str, bool] = Field(..., description="dependency check results")


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


class JobCancelResponse(BaseModel):
    job_id: UUID = Field(..., description="job ID")
    cancel_requested: bool = Field(..., description="cancel flag written")
    status: JobStatus = Field(..., description="current job status")
