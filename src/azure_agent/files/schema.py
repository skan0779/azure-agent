from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

AgentFileRole = Literal["upload", "artifact"]
SandboxSnapshot = dict[str, list[float | int]]


@dataclass(frozen=True, slots=True)
class AgentFileCreate:
    file_id: str
    user_id: str
    thread_id: str
    job_id: str | None
    role: AgentFileRole
    blob_path: str
    sandbox_path: str
    filename: str
    mime_type: str | None
    size: int


@dataclass(frozen=True, slots=True)
class AgentFile:
    file_id: str
    user_id: str
    thread_id: str
    job_id: str | None
    role: AgentFileRole
    blob_path: str
    sandbox_path: str
    filename: str
    mime_type: str | None
    size: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class SandboxSession:
    user_id: str
    thread_id: str
    session_marker: str
    last_snapshot: SandboxSnapshot = field(default_factory=dict)
    updated_at: datetime | None = None


__all__ = [
    "AgentFile",
    "AgentFileCreate",
    "AgentFileRole",
    "SandboxSession",
    "SandboxSnapshot",
]
