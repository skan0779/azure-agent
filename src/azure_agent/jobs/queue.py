from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from redis.asyncio.cluster import RedisCluster

from azure_agent.api.schema import JobStatus


async def _reenqueue_existing_job(
    redis_client: RedisCluster,
    *,
    existing_job_id: str,
    existing_job: dict[str, Any],
    request_stream_key: str,
    request_stream_maxlen: int | None,
    idempotency_key: str | None,
    job_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    reenqueue_fields: dict[str, str] = {
        "job_id": existing_job_id,
        "thread_id": str(existing_job.get("thread_id", "")),
        "user_id": str(existing_job.get("user_id", "")),
        "created_at": str(existing_job.get("created_at", "")),
    }
    if idempotency_key:
        reenqueue_fields["idempotency_key"] = idempotency_key

    try:
        reenqueue_id_raw = await redis_client.xadd(
            request_stream_key,
            fields=reenqueue_fields,
            maxlen=request_stream_maxlen,
            approximate=True,
        )
        reenqueue_id = (
            reenqueue_id_raw.decode("utf-8", errors="replace")
            if isinstance(reenqueue_id_raw, bytes)
            else str(reenqueue_id_raw)
        )
        await redis_client.hset(
            f"job:{existing_job_id}",
            mapping={
                "enqueue_state": "enqueued",
                "request_stream_id": reenqueue_id,
            },
        )
        if job_ttl_seconds:
            await redis_client.expire(f"job:{existing_job_id}", job_ttl_seconds)
        existing_job["enqueue_state"] = "enqueued"
        existing_job["request_stream_id"] = reenqueue_id
    except Exception as exc:
        failed_at = datetime.now(timezone.utc).isoformat()
        await redis_client.hset(
            f"job:{existing_job_id}",
            mapping={
                "enqueue_state": "enqueue_failed",
                "enqueue_error": str(exc),
                "enqueue_failed_at": failed_at,
            },
        )
        if job_ttl_seconds:
            await redis_client.expire(f"job:{existing_job_id}", job_ttl_seconds)
        existing_job["enqueue_state"] = "enqueue_failed"
        existing_job["enqueue_error"] = str(exc)
        existing_job["enqueue_failed_at"] = failed_at

    return existing_job


async def create_job(
    redis_client: RedisCluster,
    *,
    thread_id: str,
    user_id: str,
    user_query: str,
    idempotency_key: str | None = None,
    metadata: dict[str, Any] | None = None,
    request_stream_key: str = "agent:requests",
    request_stream_maxlen: int | None = 100_000,
    job_ttl_seconds: int | None = None,
    idempotency_ttl_seconds: int = 60 * 60 * 24,
) -> dict[str, Any]:
    """
    Create and enqueue a job. (Idempotent if `idempotency_key` is provided.)

    Args:
        redis_client (RedisCluster): Redis client instance
        thread_id (str): Thread ID for the job
        user_id (str): User ID for the job
        user_query (str): User query or input for the job
        idempotency_key (str | None): Optional idempotency key for deduplication
        metadata (dict[str, Any] | None): Optional metadata to store with the job
        request_stream_key (str): Redis stream key for enqueuing job requests
        request_stream_maxlen (int | None): Maximum length of the Redis stream
        job_ttl_seconds (int | None): Optional TTL for the job hash in seconds
        idempotency_ttl_seconds (int): TTL for the idempotency key in seconds
    Returns:
        dict (dict[str, Any]): A dictionary containing job details and creation status
    """
    job_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    job_hash_key = f"job:{job_id}"
    metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else ""

    idempo_key: str | None = None
    if idempotency_key:
        idempo_key = f"idempo:{user_id}:{idempotency_key}"
        existing_raw = await redis_client.get(idempo_key)
        if existing_raw is not None:
            existing_job_id = (
                existing_raw.decode("utf-8", errors="replace")
                if isinstance(existing_raw, bytes)
                else str(existing_raw)
            )
            existing_job = await get_job(redis_client, existing_job_id)
            if existing_job is None:
                await redis_client.delete(idempo_key)
            else:
                if existing_job.get("enqueue_state") == "enqueue_failed":
                    existing_job = await _reenqueue_existing_job(
                        redis_client,
                        existing_job_id=existing_job_id,
                        existing_job=existing_job,
                        request_stream_key=request_stream_key,
                        request_stream_maxlen=request_stream_maxlen,
                        idempotency_key=idempotency_key,
                        job_ttl_seconds=job_ttl_seconds,
                    )
                existing_job["created"] = False
                return existing_job

        claimed = await redis_client.set(
            idempo_key,
            job_id,
            ex=idempotency_ttl_seconds,
            nx=True,
        )
        if not claimed:
            winner_job_id: str | None = None
            for _ in range(5):
                winner_raw = await redis_client.get(idempo_key)
                winner_job_id = (
                    winner_raw.decode("utf-8", errors="replace")
                    if isinstance(winner_raw, bytes)
                    else str(winner_raw)
                ) if winner_raw is not None else None

                if winner_job_id:
                    winner_job = await get_job(redis_client, winner_job_id)
                    if winner_job is not None:
                        winner_job["created"] = False
                        return winner_job
                await asyncio.sleep(0.05)

            if winner_job_id:
                return {
                    "job_id": winner_job_id,
                    "status": "queued",
                    "request_stream_id": None,
                    "created": False,
                }
            raise RuntimeError(
                "[jobs.queue] Failed to resolve idempotency winner."
            )

    job_fields: dict[str, str] = {
        "job_id": job_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "user_query": user_query,
        "status": "queued",
        "cancel_requested": "0",
        "created_at": created_at,
        "enqueue_state": "pending",
    }
    if idempotency_key:
        job_fields["idempotency_key"] = idempotency_key
    if metadata_json:
        job_fields["metadata"] = metadata_json

    await redis_client.hset(job_hash_key, mapping=job_fields)
    if job_ttl_seconds:
        await redis_client.expire(job_hash_key, job_ttl_seconds)

    request_fields: dict[str, str] = {
        "job_id": job_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "created_at": created_at,
    }
    if idempotency_key:
        request_fields["idempotency_key"] = idempotency_key

    try:
        request_stream_id_raw = await redis_client.xadd(
            request_stream_key,
            fields=request_fields,
            maxlen=request_stream_maxlen,
            approximate=True,
        )
        request_stream_id = (
            request_stream_id_raw.decode("utf-8", errors="replace")
            if isinstance(request_stream_id_raw, bytes)
            else str(request_stream_id_raw)
        )
        await redis_client.hset(
            job_hash_key,
            mapping={
                "enqueue_state": "enqueued",
                "request_stream_id": request_stream_id,
            },
        )
    except Exception as exc:
        await redis_client.hset(
            job_hash_key,
            mapping={
                "enqueue_state": "enqueue_failed",
                "enqueue_error": str(exc),
                "enqueue_failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        request_stream_id = None

    response: dict[str, Any] = {
        "job_id": job_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "user_query": user_query,
        "status": "queued",
        "cancel_requested": False,
        "created_at": created_at,
        "request_stream_id": request_stream_id,
        "created": True,
        "enqueue_state": "enqueued" if request_stream_id else "enqueue_failed",
    }
    if idempotency_key:
        response["idempotency_key"] = idempotency_key
    if metadata is not None:
        response["metadata"] = metadata
    return response


async def get_job(redis_client: RedisCluster, job_id: str) -> dict[str, Any] | None:
    """
    Read job hash.
    Args:
        redis_client (RedisCluster): Redis client instance
        job_id (str): Job ID
    Returns:
        dict[str, Any] | None: Job data or None if not found
    """
    values = await redis_client.hgetall(f"job:{job_id}")
    if not values:
        return None

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in values.items():
        key = (
            raw_key.decode("utf-8", errors="replace")
            if isinstance(raw_key, bytes)
            else str(raw_key)
        )
        value = (
            raw_value.decode("utf-8", errors="replace")
            if isinstance(raw_value, bytes)
            else str(raw_value)
        )
        normalized[key] = value

    normalized["cancel_requested"] = normalized.get("cancel_requested", "0") == "1"

    metadata_raw = normalized.get("metadata")
    if metadata_raw:
        try:
            normalized["metadata"] = json.loads(metadata_raw)
        except json.JSONDecodeError:
            pass
    return normalized


async def get_job_by_idempotency(
    redis_client: RedisCluster,
    *,
    user_id: str,
    idempotency_key: str,
    request_stream_key: str = "agent:requests",
    request_stream_maxlen: int | None = 100_000,
    job_ttl_seconds: int | None = None,
) -> dict[str, Any] | None:
    """
    Resolve an existing job by idempotency key without creating a new job.
    """
    idempo_key = f"idempo:{user_id}:{idempotency_key}"
    existing_raw = await redis_client.get(idempo_key)
    if existing_raw is None:
        return None

    existing_job_id = (
        existing_raw.decode("utf-8", errors="replace")
        if isinstance(existing_raw, bytes)
        else str(existing_raw)
    )
    existing_job = await get_job(redis_client, existing_job_id)
    if existing_job is None:
        await redis_client.delete(idempo_key)
        return None

    if existing_job.get("enqueue_state") == "enqueue_failed":
        existing_job = await _reenqueue_existing_job(
            redis_client,
            existing_job_id=existing_job_id,
            existing_job=existing_job,
            request_stream_key=request_stream_key,
            request_stream_maxlen=request_stream_maxlen,
            idempotency_key=idempotency_key,
            job_ttl_seconds=job_ttl_seconds,
        )

    existing_job["created"] = False
    return existing_job


async def patch_job(
    redis_client: RedisCluster,
    job_id: str,
    *,
    status: JobStatus | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    started: bool = False,
    finished: bool = False,
    ttl_seconds: int | None = None,
) -> bool:
    """
    Patch job hash.
    Args:
        redis_client (RedisCluster): Redis client instance
        job_id (str): Job ID
        status (Literal["queued", "running", "completed", "failed", "cancelled"] | None): New status
        error (str | None): Error message if any
        metadata (dict[str, Any] | None): Metadata to update
        started (bool): Whether to set started_at timestamp
        finished (bool): Whether to set finished_at timestamp
        ttl_seconds (int | None): Optional TTL refresh for the job hash
    Returns:
        bool: True if job existed and was patched, False otherwise
    """
    key = f"job:{job_id}"
    exists = await redis_client.exists(key)
    if not exists:
        return False

    fields: dict[str, str] = {}
    if status:
        fields["status"] = status.value
    if error is not None:
        fields["error"] = error
    if metadata is not None:
        fields["metadata"] = json.dumps(metadata, ensure_ascii=False)
    if started:
        fields["started_at"] = datetime.now(timezone.utc).isoformat()
    if finished:
        fields["finished_at"] = datetime.now(timezone.utc).isoformat()

    if fields:
        await redis_client.hset(key, mapping=fields)
    if ttl_seconds:
        await redis_client.expire(key, ttl_seconds)
    return True


async def cancel_job(
    redis_client: RedisCluster,
    job_id: str,
    *,
    ttl_seconds: int | None = None,
) -> bool:
    """
    Set `cancel_requested` flag in job hash.
    Args:
        redis_client (RedisCluster): Redis client instance
        job_id (str): Job ID
    Returns:
        bool: True if job existed
    """
    key = f"job:{job_id}"
    exists = await redis_client.exists(key)
    if not exists:
        return False
    await redis_client.hset(key, mapping={"cancel_requested": "1"})
    if ttl_seconds:
        await redis_client.expire(key, ttl_seconds)
    return True


async def append_event(
    redis_client: RedisCluster,
    job_id: str,
    event: Mapping[str, Any],
    *,
    ttl: int | None = None,
) -> str:
    """
    Append event to `results:{job_id}` stream.
    Args:
        redis_client (RedisCluster): Redis client instance
        job_id (str): Job ID
        event (Mapping[str, Any]): Event data to append
        ttl (int | None): Optional TTL for the stream in seconds
    Returns:
        str: Event ID
    """
    stream_key = f"results:{job_id}"
    fields: dict[str, str] = {}
    for key, value in event.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            fields[str(key)] = str(value)
        else:
            try:
                fields[str(key)] = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                fields[str(key)] = str(value)

    if not fields:
        fields = {"payload": "{}"}
    fields["created_at"] = datetime.now(timezone.utc).isoformat()

    event_id = await redis_client.xadd(stream_key, fields=fields)
    if ttl:
        await redis_client.expire(stream_key, ttl)
    return (
        event_id.decode("utf-8", errors="replace")
        if isinstance(event_id, bytes)
        else str(event_id)
    )


async def read_events(
    redis_client: RedisCluster,
    job_id: str,
    *,
    last_id: str,
    block_ms: int = 15000,
    count: int = 100,
) -> list[dict[str, Any]]:
    """
    Read events from `results:{job_id}` stream.
    Args:
        redis_client (RedisCluster): Redis client instance
        job_id (str): Job ID
        last_id (str): Last event ID received (use "0-0" to read from the beginning)
        block_ms (int): Milliseconds to block if no new events
        count (int): Maximum number of events to read
    Returns:
        list (list[dict[str, Any]]): List of events with 'id' and 'fields'
    """
    records = await redis_client.xread(
        streams={f"results:{job_id}": last_id},
        block=block_ms,
        count=count,
    )
    if not records:
        return []

    events: list[dict[str, Any]] = []
    for _, items in records:
        for event_id, fields in items:
            normalized_fields: dict[str, Any] = {}
            for raw_key, raw_value in fields.items():
                key = (
                    raw_key.decode("utf-8", errors="replace")
                    if isinstance(raw_key, bytes)
                    else str(raw_key)
                )
                value = (
                    raw_value.decode("utf-8", errors="replace")
                    if isinstance(raw_value, bytes)
                    else str(raw_value)
                )
                if key in {"metadata", "payload"}:
                    try:
                        normalized_fields[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                normalized_fields[key] = value
            normalized_event_id = (
                event_id.decode("utf-8", errors="replace")
                if isinstance(event_id, bytes)
                else str(event_id)
            )
            events.append({"id": normalized_event_id, "fields": normalized_fields})
    return events
