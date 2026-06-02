<h1 align="center">Session Manager</h1>

<p align="center">
  Runtime session state management per thread_id
</p>

---

## 1. Resource
> Use same `Azure Managed Redis` (Redis Stream) instance as Job Queue

- `session:{thread_id}:meta` : session metadata storage (`thread_id`, `user_id`, `status`, `created_at`, `last_seen_at`, `last_job_id`, `active_job_id`)
- `session:{thread_id}:active_job` : active job ID association for the session
- `session:{thread_id}:lock` : TTL lock for in-progress worker execution

---

## 2. Feature
> Session Manager main features

### Main Features
- session metadata creation and update
- `thread_id` ownership validation
- active job lookup, assignment, and cleanup
- reservation token allocation before job creation
- processing lock acquisition, refresh, and release
- worker heartbeat
- session cleanup after job completion

### Error Code
- `403 session_ownership_error` : use of a `thread_id` owned by another user
- `409 session_conflict` : existing active job on the same `thread_id`
- `409 idempotency_conflict` : existing binding of the same `idempotency_key` to a different job context

---

## 3. Runtime Flow
> Runtime Flow with Job/Worker and Session Manager

### Job Create
1. backend request to `POST /agent/api/jobs` with header (`thread_id`,`user_query`, `X-User-Id`)
2. ownership validation and active job state check by Session Manager
3. `pending:<uuid>` reservation creation when no active job exists
4. actual job creation in the job queue
5. reservation token binding to the actual `job_id`

### Job Run
1. job read by the worker
2. acquisition of `session:{thread_id}:lock`
3. session status update to `running`
4. heartbeat and lock refresh during execution
5. active job and lock cleanup after completion, failure, or cancellation

---

## 4. Policy
> Session Manager Policy

### Session Owner Policy
- single-`user_id` binding for the same `thread_id`
- `X-User-Id` header as the source of truth for caller identity
- `403` response for requests using a `thread_id` already bound to a different `user_id`

### Concurrency Policy
- no concurrent active jobs for the same `thread_id`
- `409 session_conflict` response when an active job already exists
- `pending:<uuid>` reservation write before job creation and replacement with the real `job_id` after successful creation

### Idempotency Policy
- existing job reuse for the same `user_id + idempotency_key` combination
- `409 idempotency_conflict` response when the existing job's `thread_id` or `user_id` does not match the current request

### Runtime Session Policy
- automatic runtime session creation for a newly seen `thread_id`
- default session TTL of `1 hour`
- default reservation TTL of `5 minutes`
- default processing lock TTL of `90 seconds`
- 15-second heartbeat and lock refresh interval during job execution
- worker-side execution stop after processing lock loss

### Cleanup Policy
- active job and lock cleanup after job completion, failure, or cancellation
- session metadata retention until TTL expiration
- job and event TTL behavior following the existing job queue policy
