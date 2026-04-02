PENDING_JOB_PREFIX = "pending:"


def session_meta_key(thread_id: str) -> str:
    return f"session:{thread_id}:meta"


def session_active_job_key(thread_id: str) -> str:
    return f"session:{thread_id}:active_job"


def session_lock_key(thread_id: str) -> str:
    return f"session:{thread_id}:lock"


def is_pending_job_ref(value: str | None) -> bool:
    return bool(value and value.startswith(PENDING_JOB_PREFIX))
