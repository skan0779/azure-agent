from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class SessionStatus(str, Enum):
    idle = "idle"
    queued = "queued"
    running = "running"
    closed = "closed"


@dataclass(slots=True)
class SessionMeta:
    thread_id: str
    user_id: str
    status: str
    created_at: str
    last_seen_at: str
    last_job_id: str | None = None
    active_job_id: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, str]) -> "SessionMeta":
        return cls(
            thread_id=str(data.get("thread_id", "")),
            user_id=str(data.get("user_id", "")),
            status=str(data.get("status", SessionStatus.idle.value)),
            created_at=str(data.get("created_at", "")),
            last_seen_at=str(data.get("last_seen_at", "")),
            last_job_id=(
                str(data["last_job_id"])
                if data.get("last_job_id") not in {None, ""}
                else None
            ),
            active_job_id=(
                str(data["active_job_id"])
                if data.get("active_job_id") not in {None, ""}
                else None
            ),
        )
