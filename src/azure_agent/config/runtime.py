from __future__ import annotations

import os
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class JobRuntimeConfig:
    job_ttl_seconds: int
    event_ttl_seconds: int
    idempotency_ttl_seconds: int


@dataclass(frozen=True, slots=True)
class SessionRuntimeConfig:
    session_ttl_seconds: int
    reservation_ttl_seconds: int
    lock_ttl_seconds: int


@dataclass(frozen=True, slots=True)
class WorkerRuntimeConfig:
    heartbeat_interval_seconds: int
    pending_claim_idle_ms: int
    pending_claim_count: int
    read_block_ms: int
    read_count: int


@dataclass(frozen=True, slots=True)
class ApiRuntimeConfig:
    sse_max_connection_seconds: int


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    api: ApiRuntimeConfig
    job: JobRuntimeConfig
    session: SessionRuntimeConfig
    worker: WorkerRuntimeConfig


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        api=ApiRuntimeConfig(
            sse_max_connection_seconds=int(
                os.getenv("SSE_MAX_CONNECTION_SECONDS") or 60 * 10
            ),
        ),
        job=JobRuntimeConfig(
            job_ttl_seconds=int(os.getenv("JOB_TTL_SECONDS") or 60 * 60 * 24),
            event_ttl_seconds=int(os.getenv("EVENT_TTL_SECONDS") or 60 * 60 * 24),
            idempotency_ttl_seconds=int(
                os.getenv("IDEMPOTENCY_TTL_SECONDS") or 60 * 60 * 24
            ),
        ),
        session=SessionRuntimeConfig(
            session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS") or 60 * 60),
            reservation_ttl_seconds=int(
                os.getenv("SESSION_RESERVATION_TTL_SECONDS") or 60 * 5
            ),
            lock_ttl_seconds=int(os.getenv("SESSION_LOCK_TTL_SECONDS") or 90),
        ),
        worker=WorkerRuntimeConfig(
            heartbeat_interval_seconds=int(
                os.getenv("WORKER_HEARTBEAT_INTERVAL_SECONDS") or 15
            ),
            pending_claim_idle_ms=int(
                os.getenv("WORKER_PENDING_CLAIM_IDLE_MS") or 60 * 5 * 1000
            ),
            pending_claim_count=int(os.getenv("WORKER_PENDING_CLAIM_COUNT") or 2),
            read_block_ms=int(os.getenv("WORKER_READ_BLOCK_MS") or 10000),
            read_count=int(os.getenv("WORKER_READ_COUNT") or 1),
        ),
    )
