from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from azure_agent.session.exceptions import SessionConflictError, SessionOwnershipError
from azure_agent.session.keys import (
    PENDING_JOB_PREFIX,
    is_pending_job_ref,
    session_active_job_key,
    session_lock_key,
    session_meta_key,
)
from azure_agent.session.schema import SessionMeta, SessionStatus


class SessionManager:
    """Redis-backed runtime session manager."""

    def __init__(
        self,
        redis_client,
        *,
        lock_ttl_seconds: int = 90,
        session_ttl_seconds: int = 60 * 60,
        reservation_ttl_seconds: int = 60 * 5,
    ) -> None:
        self.redis_client = redis_client
        self.lock_ttl_seconds = lock_ttl_seconds
        self.session_ttl_seconds = session_ttl_seconds
        self.reservation_ttl_seconds = reservation_ttl_seconds

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def get_session(self, thread_id: str) -> SessionMeta | None:
        values = await self.redis_client.hgetall(session_meta_key(thread_id))
        if not values:
            return None
        return SessionMeta.from_mapping(values)

    async def get_active_job(self, thread_id: str) -> str | None:
        value = await self.redis_client.get(session_active_job_key(thread_id))
        if value in {None, ""}:
            return None
        return str(value)

    async def upsert_session(
        self,
        *,
        thread_id: str,
        user_id: str,
        status: SessionStatus = SessionStatus.idle,
        last_job_id: str | None = None,
        active_job_id: str | None = None,
    ) -> SessionMeta:
        existing = await self.get_session(thread_id)
        if existing is not None and existing.user_id != user_id:
            raise SessionOwnershipError(
                thread_id=thread_id,
                expected_user_id=existing.user_id,
                actual_user_id=user_id,
            )

        now = self._now()
        fields: dict[str, str] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "status": status.value,
            "created_at": existing.created_at if existing is not None else now,
            "last_seen_at": now,
        }
        if last_job_id is not None:
            fields["last_job_id"] = last_job_id
        if active_job_id is not None:
            fields["active_job_id"] = active_job_id

        meta_key = session_meta_key(thread_id)
        await self.redis_client.hset(meta_key, mapping=fields)
        await self.redis_client.expire(meta_key, self.session_ttl_seconds)

        if active_job_id is not None:
            await self.redis_client.set(
                session_active_job_key(thread_id),
                active_job_id,
                ex=self.session_ttl_seconds,
            )

        return SessionMeta.from_mapping(
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "status": status.value,
                "created_at": fields["created_at"],
                "last_seen_at": now,
                "last_job_id": last_job_id if last_job_id is not None else (
                    existing.last_job_id if existing is not None else ""
                ),
                "active_job_id": active_job_id if active_job_id is not None else (
                    existing.active_job_id if existing is not None else ""
                ),
            }
        )

    async def heartbeat(
        self,
        *,
        thread_id: str,
        user_id: str,
        status: SessionStatus,
        last_job_id: str | None = None,
        active_job_id: str | None = None,
    ) -> SessionMeta:
        return await self.upsert_session(
            thread_id=thread_id,
            user_id=user_id,
            status=status,
            last_job_id=last_job_id,
            active_job_id=active_job_id,
        )

    async def reserve_job(
        self,
        *,
        thread_id: str,
        user_id: str,
    ) -> str:
        await self.upsert_session(
            thread_id=thread_id,
            user_id=user_id,
            status=SessionStatus.idle,
        )

        reservation_id = f"{PENDING_JOB_PREFIX}{uuid4()}"
        claimed = await self.redis_client.set(
            session_active_job_key(thread_id),
            reservation_id,
            ex=self.reservation_ttl_seconds,
            nx=True,
        )
        if not claimed:
            raise SessionConflictError(
                thread_id=thread_id,
                active_job_id=await self.get_active_job(thread_id),
            )

        await self.redis_client.hset(
            session_meta_key(thread_id),
            mapping={
                "status": SessionStatus.queued.value,
                "last_seen_at": self._now(),
                "active_job_id": reservation_id,
            },
        )
        await self.redis_client.expire(
            session_meta_key(thread_id),
            self.session_ttl_seconds,
        )
        return reservation_id

    async def bind_job(
        self,
        *,
        thread_id: str,
        user_id: str,
        reservation_id: str,
        job_id: str,
        status: SessionStatus,
    ) -> None:
        meta = await self.get_session(thread_id)
        if meta is not None and meta.user_id != user_id:
            raise SessionOwnershipError(
                thread_id=thread_id,
                expected_user_id=meta.user_id,
                actual_user_id=user_id,
            )

        current = await self.get_active_job(thread_id)
        if current not in {reservation_id, job_id}:
            raise SessionConflictError(
                thread_id=thread_id,
                active_job_id=current,
            )

        await self.upsert_session(
            thread_id=thread_id,
            user_id=user_id,
            status=status,
            last_job_id=job_id,
            active_job_id=job_id,
        )

    async def clear_active_job(
        self,
        *,
        thread_id: str,
        user_id: str,
        expected_job_id: str | None = None,
        status: SessionStatus = SessionStatus.idle,
        last_job_id: str | None = None,
    ) -> bool:
        existing = await self.get_session(thread_id)
        if existing is not None and existing.user_id != user_id:
            raise SessionOwnershipError(
                thread_id=thread_id,
                expected_user_id=existing.user_id,
                actual_user_id=user_id,
            )
        current = await self.get_active_job(thread_id)
        if current is not None and expected_job_id not in {None, current}:
            return False

        meta = await self.upsert_session(
            thread_id=thread_id,
            user_id=user_id,
            status=status,
            last_job_id=last_job_id,
        )
        await self.redis_client.delete(session_active_job_key(thread_id))
        await self.redis_client.hdel(session_meta_key(thread_id), "active_job_id")
        await self.redis_client.expire(
            session_meta_key(thread_id),
            self.session_ttl_seconds,
        )
        meta.active_job_id = None
        return True

    async def acquire_processing_lock(self, *, thread_id: str, job_id: str) -> bool:
        claimed = await self.redis_client.set(
            session_lock_key(thread_id),
            job_id,
            ex=self.lock_ttl_seconds,
            nx=True,
        )
        return bool(claimed)

    async def refresh_processing_lock(self, *, thread_id: str, job_id: str) -> bool:
        current = await self.redis_client.get(session_lock_key(thread_id))
        if current != job_id:
            return False
        await self.redis_client.expire(
            session_lock_key(thread_id),
            self.lock_ttl_seconds,
        )
        return True

    async def release_processing_lock(self, *, thread_id: str, job_id: str) -> bool:
        current = await self.redis_client.get(session_lock_key(thread_id))
        if current != job_id:
            return False
        await self.redis_client.delete(session_lock_key(thread_id))
        return True

    async def mark_job_running(
        self,
        *,
        thread_id: str,
        user_id: str,
        job_id: str,
    ) -> None:
        await self.upsert_session(
            thread_id=thread_id,
            user_id=user_id,
            status=SessionStatus.running,
            last_job_id=job_id,
            active_job_id=job_id,
        )

    async def mark_job_complete(
        self,
        *,
        thread_id: str,
        user_id: str,
        job_id: str,
    ) -> None:
        await self.release_processing_lock(thread_id=thread_id, job_id=job_id)
        await self.clear_active_job(
            thread_id=thread_id,
            user_id=user_id,
            expected_job_id=job_id,
            status=SessionStatus.idle,
            last_job_id=job_id,
        )

    async def close_session(self, *, thread_id: str) -> None:
        await self.redis_client.delete(
            session_meta_key(thread_id),
            session_active_job_key(thread_id),
            session_lock_key(thread_id),
        )

    @staticmethod
    def is_pending_job_ref(value: str | None) -> bool:
        return is_pending_job_ref(value)
