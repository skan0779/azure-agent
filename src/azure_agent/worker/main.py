from __future__ import annotations

import asyncio, logging, os, signal, sys
from typing import Any

from redis.exceptions import ResponseError

from graphs.graph import LangGraphProcess
from infra.key_vault import create_secret_client
from infra.redis import close_redis_client, create_redis_stream_client
from jobs.job_queue import append_event, get_job, patch_job
from schemas.api import JobStatus

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
        self.request_stream_key = "agent:requests"
        self.consumer_group = "agent-workers"
        self.consumer_name = f"worker-{os.getpid()}"
        self.read_block_ms = 10000
        self.read_count = 1
        self.event_ttl_seconds = 86400
        self._stop = asyncio.Event()
        self.agent: LangGraphProcess | None = None
        self.redis_stream_client = None

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

        # Set LangGraph Agent
        self.agent = LangGraphProcess(
            secret_client=secret_client
        )

        # Setup LangGraph Agent
        if getattr(self.agent, "graph", None) is None:
            await self.agent.setup()

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
                    await self._process_entry(stream_id=str(stream_id), fields=fields)

    async def _process_entry(self, *, stream_id: str, fields: dict[str, Any]) -> None:
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

        job_id = str(fields.get("job_id", "")).strip()
        thread_id = str(fields.get("thread_id", "")).strip()
        user_id = str(fields.get("user_id", "")).strip()

        if not job_id:
            logger.warning("[worker.main] Missing job_id in stream event: %s", fields)
            await self.redis_stream_client.xack(
                self.request_stream_key,
                self.consumer_group,
                stream_id,
            )
            return

        # Process Job
        try:
            # Read Job
            job = await get_job(self.redis_stream_client, job_id)
            if job is None:
                logger.warning("[worker.main] Job not found: %s", job_id)
                await self.redis_stream_client.xack(
                    self.request_stream_key,
                    self.consumer_group,
                    stream_id,
                )
                return

            thread_id = thread_id or str(job.get("thread_id", ""))
            user_id = user_id or str(job.get("user_id", ""))
            user_query = str(job.get("user_query", ""))

            # Patch Job (cancelled: before start)
            if job.get("cancel_requested"):
                await patch_job(
                    self.redis_stream_client,
                    job_id,
                    status=JobStatus.cancelled,
                    finished=True,
                )
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    {"type": "cancelled", "content": "Job cancelled before start"},
                    ttl=self.event_ttl_seconds,
                )
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    {"type": "complete"},
                    ttl=self.event_ttl_seconds,
                )
                await self.redis_stream_client.xack(
                    self.request_stream_key,
                    self.consumer_group,
                    stream_id,
                )
                return

            # Patch Job (running)
            await patch_job(
                self.redis_stream_client,
                job_id,
                status=JobStatus.running,
                started=True,
            )

            async def _cancel_requested() -> bool:
                if getattr(_cancel_requested, "_cached_value", False):
                    return True

                now = asyncio.get_running_loop().time()
                if now - getattr(_cancel_requested, "_cached_at", 0.0) < 0.3:
                    return bool(getattr(_cancel_requested, "_cached_value", False))

                current_job = await get_job(self.redis_stream_client, job_id)
                value = bool(current_job is not None and current_job.get("cancel_requested"))
                _cancel_requested._cached_value = value
                _cancel_requested._cached_at = now
                return value

            # Run Job & Stream Events
            saw_complete = False
            cancelled_by_hook = False
            async for evt in self.agent.run_job(
                thread_id=thread_id,
                user_id=user_id,
                user_query=user_query,
                cancel=_cancel_requested,
            ):
                if not isinstance(evt, dict):
                    continue

                event_payload = dict(evt)
                event_type = str(event_payload.get("type", ""))
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
            if cancelled_by_hook:
                await patch_job(
                    self.redis_stream_client,
                    job_id,
                    status=JobStatus.cancelled,
                    finished=True,
                )
                await self.redis_stream_client.xack(
                    self.request_stream_key,
                    self.consumer_group,
                    stream_id,
                )
                return

            # Patch Job (completed)
            await patch_job(
                self.redis_stream_client,
                job_id,
                status=JobStatus.completed,
                finished=True,
            )
            if not saw_complete:
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    {"type": "complete"},
                    ttl=self.event_ttl_seconds,
                )

            # Acknowledge Entry
            await self.redis_stream_client.xack(
                self.request_stream_key,
                self.consumer_group,
                stream_id,
            )

        except Exception as exc:
            logger.exception("[worker.main] Failed to process job %s: %s", job_id, exc)
            try:
                await patch_job(
                    self.redis_stream_client,
                    job_id,
                    status=JobStatus.failed,
                    error=str(exc),
                    finished=True,
                )
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    {"type": "error", "content": str(exc)},
                    ttl=self.event_ttl_seconds,
                )
                await append_event(
                    self.redis_stream_client,
                    job_id,
                    {"type": "complete"},
                    ttl=self.event_ttl_seconds,
                )
            except Exception:
                pass
            finally:
                await self.redis_stream_client.xack(
                    self.request_stream_key,
                    self.consumer_group,
                    stream_id,
                )

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
