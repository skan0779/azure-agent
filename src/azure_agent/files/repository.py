from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from azure_agent.files.schema import (
    AgentFile,
    AgentFileCreate,
    AgentFileRole,
    SandboxSession,
    SandboxSnapshot,
)


class AgentFileRepository:
    def __init__(
        self,
        conn_string: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self.conn_string = conn_string
        self.pool = AsyncConnectionPool(
            conninfo=conn_string,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"connect_timeout": 5},
        )

    async def open(self) -> None:
        await self.pool.open()
        await self.pool.wait()

    async def close(self) -> None:
        await self.pool.close()

    async def insert_file_metadata(self, agent_file: AgentFileCreate) -> AgentFile:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    INSERT INTO agent_files (
                        file_id,
                        user_id,
                        thread_id,
                        job_id,
                        role,
                        blob_path,
                        sandbox_path,
                        filename,
                        mime_type,
                        size
                    )
                    VALUES (
                        %(file_id)s,
                        %(user_id)s,
                        %(thread_id)s,
                        %(job_id)s,
                        %(role)s,
                        %(blob_path)s,
                        %(sandbox_path)s,
                        %(filename)s,
                        %(mime_type)s,
                        %(size)s
                    )
                    RETURNING
                        file_id,
                        user_id,
                        thread_id,
                        job_id,
                        role,
                        blob_path,
                        sandbox_path,
                        filename,
                        mime_type,
                        size,
                        created_at
                    """,
                    {
                        "file_id": agent_file.file_id,
                        "user_id": agent_file.user_id,
                        "thread_id": agent_file.thread_id,
                        "job_id": agent_file.job_id,
                        "role": agent_file.role,
                        "blob_path": agent_file.blob_path,
                        "sandbox_path": agent_file.sandbox_path,
                        "filename": agent_file.filename,
                        "mime_type": agent_file.mime_type,
                        "size": agent_file.size,
                    },
                )
                row = await cur.fetchone()

        if row is None:
            raise RuntimeError("Failed to insert agent file metadata.")
        return _to_agent_file(row)

    async def get_file(
        self,
        *,
        file_id: str,
        user_id: str,
    ) -> AgentFile | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT
                        file_id,
                        user_id,
                        thread_id,
                        job_id,
                        role,
                        blob_path,
                        sandbox_path,
                        filename,
                        mime_type,
                        size,
                        created_at
                    FROM agent_files
                    WHERE file_id = %(file_id)s
                      AND user_id = %(user_id)s
                    """,
                    {"file_id": file_id, "user_id": user_id},
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return _to_agent_file(row)

    async def list_thread_files(
        self,
        *,
        user_id: str,
        thread_id: str,
        role: AgentFileRole | None = None,
    ) -> list[AgentFile]:
        filters = ["user_id = %(user_id)s", "thread_id = %(thread_id)s"]
        params: dict[str, Any] = {
            "user_id": user_id,
            "thread_id": thread_id,
        }
        if role is not None:
            filters.append("role = %(role)s")
            params["role"] = role

        query = f"""
            SELECT
                file_id,
                user_id,
                thread_id,
                job_id,
                role,
                blob_path,
                sandbox_path,
                filename,
                mime_type,
                size,
                created_at
            FROM agent_files
            WHERE {" AND ".join(filters)}
            ORDER BY created_at ASC, file_id ASC
        """
        return await self._fetch_agent_files(query, params)

    async def list_job_artifacts(
        self,
        *,
        user_id: str,
        thread_id: str,
        job_id: str,
    ) -> list[AgentFile]:
        return await self._fetch_agent_files(
            """
            SELECT
                file_id,
                user_id,
                thread_id,
                job_id,
                role,
                blob_path,
                sandbox_path,
                filename,
                mime_type,
                size,
                created_at
            FROM agent_files
            WHERE user_id = %(user_id)s
              AND thread_id = %(thread_id)s
              AND job_id = %(job_id)s
              AND role = 'artifact'
            ORDER BY created_at ASC, file_id ASC
            """,
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "job_id": job_id,
            },
        )

    async def _fetch_agent_files(
        self,
        query: str,
        params: Mapping[str, Any],
    ) -> list[AgentFile]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()
        return [_to_agent_file(row) for row in rows]

    # ------------------------------------------------------------------
    # Sandbox session lifecycle
    # ------------------------------------------------------------------

    async def get_sandbox_session(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> SandboxSession | None:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT user_id, thread_id, session_marker,
                           last_snapshot, updated_at
                    FROM sandbox_sessions
                    WHERE user_id = %(user_id)s
                      AND thread_id = %(thread_id)s
                    """,
                    {"user_id": user_id, "thread_id": thread_id},
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return SandboxSession(
            user_id=row["user_id"],
            thread_id=row["thread_id"],
            session_marker=row["session_marker"],
            last_snapshot=row["last_snapshot"] or {},
            updated_at=row["updated_at"],
        )

    async def upsert_sandbox_session(
        self,
        *,
        user_id: str,
        thread_id: str,
        session_marker: str,
        last_snapshot: SandboxSnapshot | None = None,
    ) -> None:
        snapshot = last_snapshot if last_snapshot is not None else {}
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO sandbox_sessions (
                        user_id, thread_id, session_marker, last_snapshot
                    )
                    VALUES (
                        %(user_id)s, %(thread_id)s,
                        %(session_marker)s, %(last_snapshot)s
                    )
                    ON CONFLICT (user_id, thread_id) DO UPDATE
                       SET session_marker = EXCLUDED.session_marker,
                           last_snapshot = EXCLUDED.last_snapshot,
                           updated_at = now()
                    """,
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "session_marker": session_marker,
                        "last_snapshot": Jsonb(snapshot),
                    },
                )

    async def update_sandbox_snapshot(
        self,
        *,
        user_id: str,
        thread_id: str,
        last_snapshot: SandboxSnapshot,
    ) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE sandbox_sessions
                       SET last_snapshot = %(last_snapshot)s,
                           updated_at = now()
                     WHERE user_id = %(user_id)s
                       AND thread_id = %(thread_id)s
                    """,
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "last_snapshot": Jsonb(last_snapshot),
                    },
                )

    # ------------------------------------------------------------------
    # File hydration tracking
    # ------------------------------------------------------------------

    async def list_unhydrated_files(
        self,
        *,
        user_id: str,
        thread_id: str,
        session_marker: str,
        role: AgentFileRole | None = None,
    ) -> list[AgentFile]:
        filters = [
            "f.user_id = %(user_id)s",
            "f.thread_id = %(thread_id)s",
            "h.file_id IS NULL",
        ]
        params: dict[str, Any] = {
            "user_id": user_id,
            "thread_id": thread_id,
            "session_marker": session_marker,
        }
        if role is not None:
            filters.append("f.role = %(role)s")
            params["role"] = role

        query = f"""
            SELECT
                f.file_id,
                f.user_id,
                f.thread_id,
                f.job_id,
                f.role,
                f.blob_path,
                f.sandbox_path,
                f.filename,
                f.mime_type,
                f.size,
                f.created_at
            FROM agent_files f
            LEFT JOIN file_hydrations h
                   ON h.file_id = f.file_id
                  AND h.session_marker = %(session_marker)s
            WHERE {" AND ".join(filters)}
            ORDER BY f.created_at ASC, f.file_id ASC
        """
        return await self._fetch_agent_files(query, params)

    async def mark_hydrated(
        self,
        *,
        file_id: str,
        session_marker: str,
    ) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO file_hydrations (file_id, session_marker)
                    VALUES (%(file_id)s, %(session_marker)s)
                    ON CONFLICT (file_id, session_marker) DO NOTHING
                    """,
                    {"file_id": file_id, "session_marker": session_marker},
                )

    async def reset_hydrations(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> None:
        """Remove hydration records for every file in the thread.

        Called when the sandbox is detected to have been recreated, so the
        next turn re-hydrates every upload from scratch.
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM file_hydrations
                     WHERE file_id IN (
                         SELECT file_id FROM agent_files
                          WHERE user_id = %(user_id)s
                            AND thread_id = %(thread_id)s
                     )
                    """,
                    {"user_id": user_id, "thread_id": thread_id},
                )

    # ------------------------------------------------------------------
    # Thread-scoped cleanup
    # ------------------------------------------------------------------

    async def list_thread_blob_paths(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> list[str]:
        """Return every blob_path stored for the given (user, thread)."""
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT blob_path
                      FROM agent_files
                     WHERE user_id = %(user_id)s
                       AND thread_id = %(thread_id)s
                    """,
                    {"user_id": user_id, "thread_id": thread_id},
                )
                rows = await cur.fetchall()
        return [row["blob_path"] for row in rows]

    async def delete_thread_metadata(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> int:
        """Delete agent_files + sandbox_sessions rows for a thread.

        ``file_hydrations`` cascades from ``agent_files`` via FK.
        Returns the number of agent_files rows removed.
        """
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        DELETE FROM agent_files
                         WHERE user_id = %(user_id)s
                           AND thread_id = %(thread_id)s
                        """,
                        {"user_id": user_id, "thread_id": thread_id},
                    )
                    deleted = cur.rowcount or 0
                    await cur.execute(
                        """
                        DELETE FROM sandbox_sessions
                         WHERE user_id = %(user_id)s
                           AND thread_id = %(thread_id)s
                        """,
                        {"user_id": user_id, "thread_id": thread_id},
                    )
        return deleted

    # ------------------------------------------------------------------
    # Filename collision resolution
    # ------------------------------------------------------------------

    async def resolve_filename_collision(
        self,
        *,
        user_id: str,
        thread_id: str,
        role: AgentFileRole,
        filename: str,
    ) -> str:
        """Return a filename that does not collide within (user, thread, role).

        Strips an existing ``" (N)"`` suffix from the input and picks the
        smallest ``N >= 1`` such that ``"<stem> (N)<ext>"`` is unused.
        Returns the original filename when there is no collision.
        """
        stem, ext = _split_stem_ext(_strip_index_suffix(filename))
        # Find every existing filename that could conflict, in one query.
        like_pattern = f"{_escape_like(stem)}%{_escape_like(ext)}"
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT filename FROM agent_files
                     WHERE user_id = %(user_id)s
                       AND thread_id = %(thread_id)s
                       AND role = %(role)s
                       AND filename LIKE %(pattern)s ESCAPE '\\'
                    """,
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "role": role,
                        "pattern": like_pattern,
                    },
                )
                rows = await cur.fetchall()

        used = {row["filename"] for row in rows}
        if filename not in used and f"{stem}{ext}" not in used:
            # No conflict with the bare filename.
            return f"{stem}{ext}"

        index = 2
        while True:
            candidate = f"{stem} ({index}){ext}"
            if candidate not in used:
                return candidate
            index += 1


_INDEX_SUFFIX_RE = re.compile(r"\s\(\d+\)$")


def _strip_index_suffix(filename: str) -> str:
    """Remove a trailing ``" (N)"`` from the stem so resolution is idempotent."""
    stem, ext = _split_stem_ext(filename)
    stripped = _INDEX_SUFFIX_RE.sub("", stem)
    return f"{stripped}{ext}"


def _split_stem_ext(filename: str) -> tuple[str, str]:
    base = posixpath.basename(filename)
    if "." in base and not base.startswith("."):
        stem, dot, ext = base.rpartition(".")
        return stem, f"{dot}{ext}"
    return base, ""


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _to_agent_file(row: Mapping[str, Any]) -> AgentFile:
    return AgentFile(
        file_id=row["file_id"],
        user_id=row["user_id"],
        thread_id=row["thread_id"],
        job_id=row["job_id"],
        role=row["role"],
        blob_path=row["blob_path"],
        sandbox_path=row["sandbox_path"],
        filename=row["filename"],
        mime_type=row["mime_type"],
        size=row["size"],
        created_at=row["created_at"],
    )


__all__ = ["AgentFileRepository"]
