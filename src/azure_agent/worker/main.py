from __future__ import annotations

import asyncio, logging, os, signal, sys, uuid
from typing import Any

from redis.exceptions import ResponseError

from langchain_azure_ai.agents.middleware import ContentSafetyViolationError

from azure_agent.config import RuntimeConfig, load_runtime_config
from azure_agent.graphs.graph import LangGraphProcess
from azure_agent.infra.key_vault import create_secret_client
from azure_agent.infra.redis import close_redis_client, create_redis_stream_client
from azure_agent.api.schema import JobStatus
from azure_agent.jobs.queue import append_event, get_job, patch_job
from azure_agent.session import SessionManager

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class JobWorker:
    """
    Job Worker that processes jobs.
    """
    def __init__(self) -> None:
        host = os.getenv("HOSTNAME") or "unknown"
        self.runtime_config: RuntimeConfig = load_runtime_config()
        self.request_stream_key = "agent:requests"
        self.consumer_group = "agent-workers"
        self.consumer_name = f"worker-{host}-{uuid.uuid4().hex[:12]}"
        self.read_block_ms = self.runtime_config.worker.read_block_ms
        self.read_count = self.runtime_config.worker.read_count
        self.pending_claim_idle_ms = self.runtime_config.worker.pending_claim_idle_ms
        self.pending_claim_count = self.runtime_config.worker.pending_claim_count
        self.pending_claim_cursor = "0-0"
        self.heartbeat_interval_seconds = (
            self.runtime_config.worker.heartbeat_interval_seconds
        )
        self.event_ttl_seconds = self.runtime_config.job.event_ttl_seconds
        self.job_ttl_seconds = self.runtime_config.job.job_ttl_seconds
        self._stop = asyncio.Event()
        self.agent: LangGraphProcess | None = None
        self.redis_stream_client = None
        self.session_manager: SessionManager | None = None

    async def _patch_job_or_raise(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        error: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        if self.redis_stream_client is None:
            raise RuntimeError("[worker.main] Redis stream client is not initialized")

        updated = await patch_job(
            self.redis_stream_client,
            job_id,
            status=status,
            error=error,
            started=started,
            finished=finished,
            ttl_seconds=self.job_ttl_seconds,
        )
        if not updated:
            raise RuntimeError(f"Job state update failed for {job_id}")

    async def _ack_entry(self, stream_id: str) -> bool:
        if self.redis_stream_client is None:
            logger.warning("[worker.main] Redis stream client unavailable during xack")
            return False

        try:
            await self.redis_stream_client.xack(
                self.request_stream_key,
                self.consumer_group,
                stream_id,
            )
            return True
        except Exception as exc:
            logger.warning(
                "[worker.main] Failed to xack stream entry %s: %s",
                stream_id,
                exc,
            )
            return False

    async def _ensure_complete_event(self, job_id: str) -> bool:
        """
        Ensure the result stream ends with a `complete` event.
        Returns True when the stream is already complete or the repair succeeds.
        """
        if self.redis_stream_client is None:
            logger.warning(
                "[worker.main] Redis stream client unavailable during complete repair"
            )
            return False

        try:
            records = await self.redis_stream_client.xrevrange(
                f"results:{job_id}",
                count=1,
            )
        except Exception:
            logger.exception(
                "[worker.main] Failed to inspect result stream for job %s",
                job_id,
            )
            return False

        if records:
            _, fields = records[0]
            raw_type = fields.get("type", b"")
            event_type = (
                raw_type.decode("utf-8", errors="replace")
                if isinstance(raw_type, bytes)
                else str(raw_type)
            )
            if event_type == "complete":
                return True

        try:
            await append_event(
                self.redis_stream_client,
                job_id,
                {"type": "complete", "ns": [], "data": None},
                ttl=self.event_ttl_seconds,
            )
            logger.info(
                "[worker.main] Repaired missing complete event for terminal job %s",
                job_id,
            )
            return True
        except Exception:
            logger.exception(
                "[worker.main] Failed to repair missing complete event for job %s",
                job_id,
            )
            return False

    async def setup(self) -> None:
        """
        Setup the JobWorker
        - set Secret Client.
        - set Redis Stream Client.
        - set LangGraph Agent.
        - setup LangGraph Agent.
        - create Redis Stream consumer group. (if not exists)
        """
        # Set Secret Client
        secret_client = create_secret_client()

        # Set Redis Stream Client
        self.redis_stream_client = create_redis_stream_client(secret_client)
        self.session_manager = SessionManager(
            self.redis_stream_client,
            lock_ttl_seconds=self.runtime_config.session.lock_ttl_seconds,
            session_ttl_seconds=self.runtime_config.session.session_ttl_seconds,
            reservation_ttl_seconds=self.runtime_config.session.reservation_ttl_seconds,
        )

        # Set LangGraph Agent
        self.agent = LangGraphProcess()

        # Setup LangGraph Agent
        if getattr(self.agent, "graph", None) is None:
            await self.agent.setup()

        # Start Store TTL Sweeper
        store = getattr(self.agent, "store", None)
        if store is not None:
            start = getattr(store, "start_ttl_sweeper", None)
            if callable(start):
                try:
                    await start(sweep_interval_minutes=60)
                    logger.info("[worker.main] Postgres TTL sweeper started")
                except Exception as exc:
                    logger.warning("[worker.main] Failed to start Postgres TTL sweeper: %s", exc)

        # Create Redis Stream Consumer Group
        await self._create_consumer_group()
        
        # Logging
        logger.info("[worker.main] Worker setup complete (stream=%s, group=%s, consumer=%s)", self.request_stream_key, self.consumer_group, self.consumer_name)

    async def _create_consumer_group(self) -> None:
        """
        Create Redis Stream Consumer Group. (if not exists)
        """
        try:
            await self.redis_stream_client.xgroup_create(
                name=self.request_stream_key,
                groupname=self.consumer_group,
                id="0-0",
                mkstream=True,
            )
            logger.info("[worker.main] Created consumer group %s on %s", self.consumer_group, self.request_stream_key)
        except ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                logger.info("[worker.main] Consumer group already exists (%s)", self.consumer_group)
            else:
                raise

    async def run(self) -> None:
        """
        Run the JobWorker to process jobs.
        - Read jobs
        - Process jobs
        """
        # Pre-checks
        if self.redis_stream_client is None:
            raise RuntimeError("[worker.main] Redis stream client is not initialized")
        if self.agent is None:
            raise RuntimeError("[worker.main] Agent is not initialized")

        # Main loop
        while not self._stop.is_set():
            reclaimed_entries = await self._claim_stale_entries()
            if reclaimed_entries:
                for stream_id, fields in reclaimed_entries:
                    await self._process_entry(
                        stream_id=stream_id,
                        fields=fields,
                        reclaimed=True,
                    )

            # Read Jobs
            try:
                records = await self.redis_stream_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.request_stream_key: ">"},
                    count=self.read_count,
                    block=self.read_block_ms,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[worker.main] xreadgroup failed: %s", exc)
                await asyncio.sleep(1.0)
                continue
            
            if not records:
                continue
            
            # Process Jobs
            for _, entries in records:
                for stream_id, fields in entries:
                    await self._process_entry(
                        stream_id=str(stream_id),
                        fields=fields,
                        reclaimed=False,
                    )

    async def _claim_stale_entries(self) -> list[tuple[str, dict[str, Any]]]:
        if self.redis_stream_client is None:
            return []

        try:
            claimed = await self.redis_stream_client.xautoclaim(
                name=self.request_stream_key,
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                min_idle_time=self.pending_claim_idle_ms,
                start_id=self.pending_claim_cursor,
                count=self.pending_claim_count,
            )
        except ResponseError as exc:
            logger.warning("[worker.main] xautoclaim failed: %s", exc)
            return []
        except Exception as exc:
            logger.warning("[worker.main] Failed to reclaim stale entries: %s", exc)
            return []

        if not isinstance(claimed, (list, tuple)) or not claimed:
            return []

        next_cursor = str(claimed[0]) if claimed[0] else "0-0"
        raw_entries = claimed[1] if len(claimed) > 1 else []
        self.pending_claim_cursor = next_cursor

        reclaimed_entries: list[tuple[str, dict[str, Any]]] = []
        for entry_id, fields in raw_entries:
            reclaimed_entries.append((str(entry_id), fields))

        if reclaimed_entries:
            logger.info(
                "[worker.main] Reclaimed %s stale pending entr%s from stream %s",
                len(reclaimed_entries),
                "y" if len(reclaimed_entries) == 1 else "ies",
                self.request_stream_key,
            )

        return reclaimed_entries

    async def _process_entry(
        self,
        *,
        stream_id: str,
        fields: dict[str, Any],
        reclaimed: bool = False,
    ) -> None:
        """
        Process a single job.
        - Read Job
        - Patch Job (cancelled: before start)
        - Patch Job (running)
        - Run Job & Stream Events
        - Patch Job (cancelled: by hook)
        - Patch Job (completed)
        - Acknowledge Entry
        Args:
            stream_id: Redis Stream Entry ID
            fields: Redis Stream Entry Fields
        """
        # Pre-checks
        if self.redis_stream_client is None:
            return
        if self.agent is None:
            return
        if self.session_manager is None:
            return

        job_id = str(fields.get("job_id", "")).strip()
        thread_id = str(fields.get("thread_id", "")).strip()
        user_id = str(fields.get("user_id", "")).strip()
        heartbeat_task: asyncio.Task | None = None
        heartbeat_stop = asyncio.Event()
        lock_lost = asyncio.Event()
        lock_acquired = False
        should_fail_job = False
        should_cleanup_session = False
        cleanup_allowed = True
        terminal_status_written = False

        if not job_id:
            logger.warning("[worker.main] Missing job_id in stream event: %s", fields)
            await self._ack_entry(stream_id)
            return

        # Process Job
        try:
            # Read Job
            job = await get_job(self.redis_stream_client, job_id)
            if job is None:
                logger.warning("[worker.main] Job not found: %s", job_id)
                await self._ack_entry(stream_id)
                return

            thread_id = thread_id or str(job.get("thread_id", ""))
            user_id = user_id or str(job.get("user_id", ""))
            user_query = str(job.get("user_query", ""))
            job_status = str(job.get("status", "")).strip()

            if job_status in {"completed", "failed", "cancelled"}:
                complete_repaired = await self._ensure_complete_event(job_id)
                if not complete_repaired:
                    logger.warning(
                        "[worker.main] Leaving terminal job pending until complete "
                        "event is repaired (job_id=%s, status=%s)",
                        job_id,
                        job_status,
                    )
                    return
                current_active_job = await self.session_manager.get_active_job(thread_id)
                if current_active_job == job_id:
                    await self.session_manager.mark_job_complete(
                        thread_id=thread_id,
                        user_id=user_id,
                        job_id=job_id,
                    )
                logger.info(
                    "[worker.main] Skip terminal job %s with status=%s",
                    job_id,
                    job_status,
                )
                await self._ack_entry(stream_id)
                return

            # Patch Job (cancelled: before start)
            if job.get("cancel_requested"):
                should_cleanup_session = True
                await self._patch_job_or_raise(
                    job_id,
                    status=JobStatus.cancelled,
                    finished=True,
                )
                terminal_status_written = True
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    {
                        "type": "cancelled",
                        "ns": [],
                        "data": {"message": "Job cancelled before start"},
                    },
                    ttl=self.event_ttl_seconds,
                )
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    {"type": "complete", "ns": [], "data": None},
                    ttl=self.event_ttl_seconds,
                )
                cleanup_allowed = await self._ack_entry(stream_id)
                if not cleanup_allowed:
                    logger.warning(
                        "[worker.main] Terminal pre-start cancel pending for reclaim "
                        "(job_id=%s)",
                        job_id,
                    )
                return

            # Patch Job (running)
            acquired = await self.session_manager.acquire_processing_lock(
                thread_id=thread_id,
                job_id=job_id,
            )
            if not acquired:
                logger.info(
                    "[worker.main] Skip duplicate or already-running job "
                    "(thread_id=%s, job_id=%s, reclaimed=%s)",
                    thread_id,
                    job_id,
                    reclaimed,
                )
                if reclaimed:
                    logger.info(
                        "[worker.main] Leaving reclaimed entry pending because "
                        "another worker still owns the session lock "
                        "(thread_id=%s, job_id=%s)",
                        thread_id,
                        job_id,
                    )
                    return
                await self._ack_entry(stream_id)
                return

            lock_acquired = True
            should_fail_job = True
            should_cleanup_session = True

            await self._patch_job_or_raise(
                job_id,
                status=JobStatus.running,
                started=True,
            )
            await self.session_manager.mark_job_running(
                thread_id=thread_id,
                user_id=user_id,
                job_id=job_id,
            )

            heartbeat_task = asyncio.create_task(
                self._heartbeat_session(
                    thread_id=thread_id,
                    user_id=user_id,
                    job_id=job_id,
                    stop_event=heartbeat_stop,
                    lock_lost_event=lock_lost,
                )
            )

            cancel_requested_cache = False
            cancel_requested_cached_at = 0.0

            async def _cancel_requested() -> bool:
                nonlocal cancel_requested_cache, cancel_requested_cached_at

                if lock_lost.is_set():
                    return True

                if cancel_requested_cache:
                    return True

                now = asyncio.get_running_loop().time()
                if now - cancel_requested_cached_at < 0.3:
                    return cancel_requested_cache

                current_job = await get_job(self.redis_stream_client, job_id)
                cancel_requested_cache = bool(
                    current_job is not None and current_job.get("cancel_requested")
                )
                cancel_requested_cached_at = now
                return cancel_requested_cache

            # Run Job & Stream Events
            saw_complete = False
            cancelled_by_hook = False
            async for evt in self.agent.run_job(
                thread_id=thread_id,
                job_id=job_id,
                user_id=user_id,
                user_query=user_query,
                cancel=_cancel_requested,
            ):
                if not isinstance(evt, dict):
                    continue

                event_payload = dict(evt)
                event_type = str(event_payload.get("type", ""))
                if lock_lost.is_set() and event_type in {"cancelled", "complete"}:
                    continue
                if event_type == "cancelled":
                    cancelled_by_hook = True
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    event_payload,
                    ttl=self.event_ttl_seconds,
                )
                if event_type == "complete":
                    saw_complete = True

            # Patch Job (cancelled: by hook)
            if lock_lost.is_set():
                raise RuntimeError("Session processing lock lost during execution")

            if cancelled_by_hook:
                await self._patch_job_or_raise(
                    job_id,
                    status=JobStatus.cancelled,
                    finished=True,
                )
                terminal_status_written = True
                cleanup_allowed = await self._ack_entry(stream_id)
                if not cleanup_allowed:
                    logger.warning(
                        "[worker.main] Terminal cancel pending for reclaim "
                        "(job_id=%s)",
                        job_id,
                    )
                return

            # Patch Job (completed)
            await self._patch_job_or_raise(
                job_id,
                status=JobStatus.completed,
                finished=True,
            )
            terminal_status_written = True
            if not saw_complete:
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    {"type": "complete", "ns": [], "data": None},
                    ttl=self.event_ttl_seconds,
                )

            # Acknowledge Entry
            cleanup_allowed = await self._ack_entry(stream_id)
            if not cleanup_allowed:
                logger.warning(
                    "[worker.main] Terminal completion pending for reclaim "
                    "(job_id=%s)",
                    job_id,
                )

        # Exception Handling
        except Exception as exc:
            logger.exception("[worker.main] Failed to process job %s: %s", job_id, exc)
            failure_persisted = False
            if terminal_status_written:
                cleanup_allowed = False
                logger.warning(
                    "[worker.main] Terminal status already written; "
                    "leaving stream entry pending for reclaim (job_id=%s)",
                    job_id,
                )
            elif should_fail_job:
                try:
                    await self._patch_job_or_raise(
                        job_id,
                        status=JobStatus.failed,
                        error=str(exc),
                        finished=True,
                    )
                    await append_event(
                        self.redis_stream_client,
                        job_id,
                        {
                            "type": "error",
                            "ns": [],
                            "data": {"message": str(exc)},
                        },
                        ttl=self.event_ttl_seconds,
                    )
                    await append_event(
                        self.redis_stream_client,
                        job_id,
                        {"type": "complete", "ns": [], "data": None},
                        ttl=self.event_ttl_seconds,
                    )
                    failure_persisted = True
                    terminal_status_written = True
                except Exception:
                    logger.exception(
                        "[worker.main] Failed to persist failure state for job %s",
                        job_id,
                    )
                cleanup_allowed = failure_persisted

                if failure_persisted:
                    cleanup_allowed = await self._ack_entry(stream_id)
                    if not cleanup_allowed:
                        logger.warning(
                            "[worker.main] Terminal failure pending for reclaim "
                            "(job_id=%s)",
                            job_id,
                        )
                else:
                    logger.warning(
                        "[worker.main] Leaving stream entry pending for retry "
                        "(job_id=%s, lock_acquired=%s)",
                        job_id,
                        lock_acquired,
                    )
            else:
                cleanup_allowed = False
                logger.warning(
                    "[worker.main] Leaving pre-lock stream entry pending for retry "
                    "(job_id=%s)",
                    job_id,
                )
        finally:
            heartbeat_stop.set()
            if heartbeat_task is not None:
                try:
                    await heartbeat_task
                except Exception:
                    pass

            if (
                self.session_manager is not None
                and thread_id
                and user_id
                and job_id
                and should_cleanup_session
                and cleanup_allowed
            ):
                try:
                    if lock_acquired:
                        await self.session_manager.mark_job_complete(
                            thread_id=thread_id,
                            user_id=user_id,
                            job_id=job_id,
                        )
                    else:
                        await self.session_manager.clear_active_job(
                            thread_id=thread_id,
                            user_id=user_id,
                            expected_job_id=job_id,
                            last_job_id=job_id,
                        )
                except Exception as exc:
                    logger.warning(
                        "[worker.main] Failed to cleanup session state for job %s: %s",
                        job_id,
                        exc,
                    )

    async def _heartbeat_session(
        self,
        *,
        thread_id: str,
        user_id: str,
        job_id: str,
        stop_event: asyncio.Event,
        lock_lost_event: asyncio.Event,
    ) -> None:
        if self.session_manager is None:
            return

        consecutive_failures = 0
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=float(self.heartbeat_interval_seconds),
                )
                return
            except asyncio.TimeoutError:
                try:
                    refreshed = await self.session_manager.refresh_processing_lock(
                        thread_id=thread_id,
                        job_id=job_id,
                    )
                    if not refreshed:
                        logger.error(
                            "[worker.main] Processing lock lost for job %s; stopping execution",
                            job_id,
                        )
                        lock_lost_event.set()
                        stop_event.set()
                        return
                    await self.session_manager.mark_job_running(
                        thread_id=thread_id,
                        user_id=user_id,
                        job_id=job_id,
                    )
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    logger.warning(
                        "[worker.main] Session heartbeat failed for job %s (%s/3): %s",
                        job_id,
                        consecutive_failures,
                        exc,
                    )
                    if consecutive_failures >= 3:
                        logger.error(
                            "[worker.main] Session heartbeat exceeded retry limit for job %s; stopping execution",
                            job_id,
                        )
                        lock_lost_event.set()
                        stop_event.set()
                        return

    async def shutdown(self) -> None:
        """Shutdown JobWorker"""
        self._stop.set()

        if self.redis_stream_client is not None:
            await close_redis_client(self.redis_stream_client)
            self.redis_stream_client = None

        if self.agent is not None:
            try:
                await self.agent.close()
            except Exception as exc:
                logger.warning("[worker.main] Agent close failed: %s", exc)
            self.agent = None

        self.session_manager = None

        logger.info("[worker.main] Worker shutdown complete")


async def _run_worker() -> None:
    worker = JobWorker()
    await worker.setup()

    loop = asyncio.get_running_loop()
    stop_called = False

    def _request_stop() -> None:
        nonlocal stop_called
        if stop_called:
            return
        stop_called = True
        logger.info("[worker.main] Stop signal received")
        asyncio.create_task(worker.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    try:
        await worker.run()
    finally:
        await worker.shutdown()


def main() -> None:
    asyncio.run(_run_worker())


if __name__ == "__main__":
    main()
