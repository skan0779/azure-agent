class SessionError(Exception):
    """Base session runtime error."""


class SessionOwnershipError(SessionError):
    def __init__(self, *, thread_id: str, expected_user_id: str, actual_user_id: str) -> None:
        self.thread_id = thread_id
        self.expected_user_id = expected_user_id
        self.actual_user_id = actual_user_id
        super().__init__(
            f"Session ownership mismatch for thread_id={thread_id}: "
            f"expected user_id={expected_user_id}, got user_id={actual_user_id}"
        )


class SessionConflictError(SessionError):
    def __init__(self, *, thread_id: str, active_job_id: str | None = None) -> None:
        self.thread_id = thread_id
        self.active_job_id = active_job_id
        super().__init__(
            f"Session conflict for thread_id={thread_id}"
            + (f", active_job_id={active_job_id}" if active_job_id else "")
        )


class SessionExpiredError(SessionError):
    def __init__(self, *, thread_id: str) -> None:
        self.thread_id = thread_id
        super().__init__(f"Session expired for thread_id={thread_id}")
