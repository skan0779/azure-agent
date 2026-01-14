import logging

from fastapi import APIRouter, HTTPException, Request

from langgraph.checkpoint.redis.base import CHECKPOINT_PREFIX, CHECKPOINT_WRITE_PREFIX
from langgraph.checkpoint.redis.util import to_storage_safe_id

from schemas.api import DeleteThreadRequest, DeleteThreadResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def _delete_checkpoint(redis_client, pattern: str, scan_count: int = 512) -> int:
    """
    Delete checkpoints matching the given pattern from Redis.

    Args:
        redis_client: Redis client instance
        pattern (str): Pattern to match keys for deletion
        scan_count (int): Number of keys to scan per iteration
    Returns:
        int: Number of keys deleted
    """
    cursor = 0
    removed = 0

    while True:
        cursor, keys = await redis_client.scan(cursor=cursor, match=pattern, count=scan_count)
        if keys:
            await redis_client.delete(*keys)
            removed += len(keys)
        if cursor == 0:
            break

    return removed


@router.post("/agent/api/delete_thread", response_model=DeleteThreadResponse, tags=["Thread"])
async def delete_thread(req: DeleteThreadRequest, request: Request):
    
    # LangGraph Agent Instance
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=500, detail="LangGraph Agent init failed")

    # Redis Client
    redis_client = getattr(agent, "redis_client", None)
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis client unavailable")

    # Thread ID
    thread_id = (req.thread_id or "").strip()
    if not thread_id:
        raise HTTPException(status_code=400, detail="Thread_ID is required")

    # Storage safe thread ID
    safe_thread_id = to_storage_safe_id(thread_id)

    try:
        # Delete checkpoints
        checkpoint_deleted = await _delete_checkpoint(redis_client, f"{CHECKPOINT_PREFIX}:{safe_thread_id}:*")

        # Delete checkpoint writes
        checkpoint_writes_deleted = await _delete_checkpoint(redis_client, f"{CHECKPOINT_WRITE_PREFIX}:{safe_thread_id}:*")

        # Delete write registries
        write_registry_deleted = await _delete_checkpoint(redis_client, f"write_keys_zset:{thread_id}:*")

    # General Exception Handling
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete thread: {exc}") from exc

    # Logging
    logger.info(
        "[thread.py] Deleted thread %s: checkpoints=%s, checkpoint_writes=%s, write_registries=%s",
        thread_id,
        checkpoint_deleted,
        checkpoint_writes_deleted,
        write_registry_deleted,
    )

    return DeleteThreadResponse(
        thread_id=thread_id,
        checkpoints=checkpoint_deleted,
        checkpoint_writes=checkpoint_writes_deleted,
        write_registries=write_registry_deleted,
    )
