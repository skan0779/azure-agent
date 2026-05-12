from __future__ import annotations

import logging
import mimetypes
import posixpath
import shlex
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import ContainerClient
from langchain.agents.middleware import AgentMiddleware
from langchain_azure_dynamic_sessions.backends import SessionsBashBackend
from langgraph.config import get_stream_writer

from azure_agent.files import (
    AgentFile,
    AgentFileCreate,
    AgentFileRepository,
    SandboxSnapshot,
)

logger = logging.getLogger(__name__)

SANDBOX_DIR = "/mnt/data"
SENTINEL_PATH = "/mnt/data/.session_marker"


class SessionsFileSyncMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        pool_management_endpoint: str,
        blob_container_client: ContainerClient,
        file_repository: AgentFileRepository,
    ) -> None:
        super().__init__()
        self.pool_management_endpoint = pool_management_endpoint
        self.blob_container_client = blob_container_client
        self.file_repository = file_repository

    async def abefore_agent(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        context = runtime.context
        backend = self._make_backend(context)

        marker = await self._ensure_session_marker(
            backend=backend,
            user_id=context.user_id,
            thread_id=context.thread_id,
        )

        pending = await self.file_repository.list_unhydrated_files(
            user_id=context.user_id,
            thread_id=context.thread_id,
            session_marker=marker,
            role="upload",
        )
        for file in pending:
            await self._hydrate_file(backend=backend, file=file)
            await self.file_repository.mark_hydrated(
                file_id=file.file_id,
                session_marker=marker,
            )

        snapshot = await self._scan_snapshot(backend)
        await self.file_repository.update_sandbox_snapshot(
            user_id=context.user_id,
            thread_id=context.thread_id,
            last_snapshot=snapshot,
        )
        return None

    async def aafter_agent(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        context = runtime.context
        backend = self._make_backend(context)

        session = await self.file_repository.get_sandbox_session(
            user_id=context.user_id,
            thread_id=context.thread_id,
        )
        before: SandboxSnapshot = session.last_snapshot if session else {}
        after = await self._scan_snapshot(backend)

        upload_paths = {
            file.sandbox_path
            for file in await self.file_repository.list_thread_files(
                user_id=context.user_id,
                thread_id=context.thread_id,
                role="upload",
            )
        }

        new_artifacts = [
            path
            for path, meta in after.items()
            if path not in upload_paths
            and not posixpath.basename(path).startswith(".")
            and before.get(path) != meta
        ]

        try:
            writer = get_stream_writer()
        except Exception:
            writer = None

        for sandbox_path in new_artifacts:
            try:
                await self._collect_artifact(
                    backend=backend,
                    user_id=context.user_id,
                    thread_id=context.thread_id,
                    job_id=context.job_id,
                    sandbox_path=sandbox_path,
                    writer=writer,
                )
            except Exception:
                logger.exception(
                    "[azure_dynamic_session.py] Failed to collect artifact %s",
                    sandbox_path,
                )

        await self.file_repository.update_sandbox_snapshot(
            user_id=context.user_id,
            thread_id=context.thread_id,
            last_snapshot=after,
        )
        return None

    def _make_backend(self, context: Any) -> SessionsBashBackend:
        return SessionsBashBackend(
            pool_management_endpoint=self.pool_management_endpoint,
            session_id=f"sandbox-{context.user_id}-{context.thread_id}",
        )

    async def _ensure_session_marker(
        self,
        *,
        backend: SessionsBashBackend,
        user_id: str,
        thread_id: str,
    ) -> str:
        sandbox_marker = await self._read_sentinel(backend)
        session = await self.file_repository.get_sandbox_session(
            user_id=user_id,
            thread_id=thread_id,
        )
        db_marker = session.session_marker if session else None

        if sandbox_marker is not None and sandbox_marker == db_marker:
            return sandbox_marker

        # Sandbox is fresh (or recreated) — rotate marker and invalidate cache.
        new_marker = uuid4().hex
        await self._write_sentinel(backend, new_marker)
        await self.file_repository.reset_hydrations(
            user_id=user_id,
            thread_id=thread_id,
        )
        await self.file_repository.upsert_sandbox_session(
            user_id=user_id,
            thread_id=thread_id,
            session_marker=new_marker,
            last_snapshot={},
        )
        logger.info(
            "[azure_dynamic_session.py] Sandbox session marker rotated "
            "(thread_id=%s, previous=%s)",
            thread_id,
            sandbox_marker,
        )
        return new_marker

    async def _read_sentinel(
        self,
        backend: SessionsBashBackend,
    ) -> str | None:
        result = await backend.aexecute(
            f"cat {shlex.quote(SENTINEL_PATH)} 2>/dev/null || true"
        )
        marker = result.output.strip()
        return marker or None

    async def _write_sentinel(
        self,
        backend: SessionsBashBackend,
        marker: str,
    ) -> None:
        command = (
            f"mkdir -p {shlex.quote(SANDBOX_DIR)} && "
            f"printf %s {shlex.quote(marker)} > {shlex.quote(SENTINEL_PATH)}"
        )
        result = await backend.aexecute(command)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to write sandbox sentinel: {result.output}"
            )

    async def _scan_snapshot(
        self,
        backend: SessionsBashBackend,
    ) -> SandboxSnapshot:
        # %T@ = mtime as epoch seconds (with fractional part), %s = size in bytes.
        command = (
            f"find {shlex.quote(SANDBOX_DIR)} -type f "
            "-printf '%p\\t%T@\\t%s\\n' 2>/dev/null || true"
        )
        result = await backend.aexecute(command)
        snapshot: SandboxSnapshot = {}
        for line in result.output.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            path, mtime_str, size_str = parts
            try:
                snapshot[path] = [float(mtime_str), int(size_str)]
            except ValueError:
                continue
        return snapshot

    async def _hydrate_file(
        self,
        *,
        backend: SessionsBashBackend,
        file: AgentFile,
    ) -> None:
        blob_client = self.blob_container_client.get_blob_client(file.blob_path)
        stream = await blob_client.download_blob()
        content = await stream.readall()
        await self._upload_to_sandbox_path(
            backend=backend,
            path=file.sandbox_path,
            content=content,
        )

    async def _collect_artifact(
        self,
        *,
        backend: SessionsBashBackend,
        user_id: str,
        thread_id: str,
        job_id: str,
        sandbox_path: str,
        writer: Any = None,
    ) -> None:
        original_filename = posixpath.basename(sandbox_path)
        filename = await self.file_repository.resolve_filename_collision(
            user_id=user_id,
            thread_id=thread_id,
            role="artifact",
            filename=original_filename,
        )
        content = await self._download_from_sandbox_path(backend, sandbox_path)
        file_id = str(uuid4())
        blob_path = f"{user_id}/{thread_id}/artifacts/{job_id}/{file_id}"
        mime_type, _ = mimetypes.guess_type(filename)

        blob_client = self.blob_container_client.get_blob_client(blob_path)
        await blob_client.upload_blob(
            data=content,
            overwrite=False,
            content_settings=ContentSettings(content_type=mime_type),
        )
        stored = await self.file_repository.insert_file_metadata(
            AgentFileCreate(
                file_id=file_id,
                user_id=user_id,
                thread_id=thread_id,
                job_id=job_id,
                role="artifact",
                blob_path=blob_path,
                sandbox_path=sandbox_path,
                filename=filename,
                mime_type=mime_type,
                size=len(content),
            )
        )

        if writer is not None:
            try:
                writer({
                    "event": "artifact_created",
                    "file_id": stored.file_id,
                    "thread_id": stored.thread_id,
                    "job_id": stored.job_id,
                    "role": stored.role,
                    "filename": stored.filename,
                    "mime_type": stored.mime_type,
                    "size": stored.size,
                    "sandbox_path": stored.sandbox_path,
                    "download_url": f"/agent/api/files/{stored.file_id}/download",
                    "created_at": (
                        stored.created_at.isoformat()
                        if stored.created_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                })
            except Exception:
                logger.exception(
                    "[azure_dynamic_session.py] Failed to emit artifact_created event for %s",
                    stored.file_id,
                )

    async def _upload_to_sandbox_path(
        self,
        *,
        backend: SessionsBashBackend,
        path: str,
        content: bytes,
    ) -> None:
        tmp_name = f".hydrate-{uuid4()}"
        responses = await backend.aupload_files([(tmp_name, content)])
        if responses[0].error is not None:
            raise RuntimeError(f"Failed to upload sandbox file: {path}")

        tmp_path = posixpath.join(SANDBOX_DIR, tmp_name)
        command = (
            f"mkdir -p {shlex.quote(posixpath.dirname(path))} && "
            f"mv {shlex.quote(tmp_path)} {shlex.quote(path)}"
        )
        result = await backend.aexecute(command)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to move sandbox file to {path}: {result.output}"
            )

    async def _download_from_sandbox_path(
        self,
        backend: SessionsBashBackend,
        path: str,
    ) -> bytes:
        tmp_name = f".download-{uuid4()}"
        tmp_path = posixpath.join(SANDBOX_DIR, tmp_name)
        copy_result = await backend.aexecute(
            f"cp {shlex.quote(path)} {shlex.quote(tmp_path)}"
        )
        if copy_result.exit_code != 0:
            raise RuntimeError(
                f"Failed to stage sandbox file {path}: {copy_result.output}"
            )

        try:
            responses = await backend.adownload_files([tmp_name])
            if responses[0].error is not None or responses[0].content is None:
                raise RuntimeError(f"Failed to download sandbox file: {path}")
            return responses[0].content
        finally:
            await backend.aexecute(f"rm -f {shlex.quote(tmp_path)}")
