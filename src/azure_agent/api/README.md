<p align="center">
  <img src="/docs/icons/FastAPI.svg" height="72" alt="FastAPI" />
</p>

<h1 align="center">API</h1>

<p align="center">
  FastAPI & Redis Stream based API
</p>

---

## 1. Overview
> Asynchronous Job API for the Agent service (validates the request, creates a job, exposes SSE events, and delegates execution to the worker process)

- `FastAPI` for HTTP routing
- `Redis Stream` for job queueing and SSE event replay
- `SessionManager` for per-`thread_id` runtime coordination
- `Worker` for background job execution

---

## 2. Endpoints
> Main API endpoints

| Endpoint | Detail |
| --- | --- |
| `GET /agent/api/ping` | liveness check |
| `GET /agent/api/health` | Readiness check (`runtime_config`, `session_manager`, `redis_client`, `redis_ping`) |
| `POST /agent/api/jobs` | Create a new job |
| `GET /agent/api/jobs/{job_id}` | Fetch current job status |
| `GET /agent/api/jobs/{job_id}/events` | Subscribe job events (SSE) |
| `POST /agent/api/jobs/{job_id}/cancel` | Request cancellation for job |

---

## 3. API Usage
> Backend usage flow by endpoint

### 3.1 `POST /agent/api/jobs`
> Create a new async job

Request Header:
```http
X-User-Id: user-123
```

Request Body:
```json
{
  "thread_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
  "user_query": "Hello?",
  "idempotency_key": "req-20260211-0001"
}
```

Response:
```json
{
  "job_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
  "status": "queued",
  "status_url": "/agent/api/jobs/44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
  "events_url": "/agent/api/jobs/44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6/events",
  "cancel_url": "/agent/api/jobs/44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6/cancel"
}
```

Errors:
| Status | Code | Detail |
| --- | --- | --- |
| `401` | `missing_user_identity` | `X-User-Id` header is missing |
| `403` | `session_ownership_error` | `thread_id` is already bound to another user |
| `409` | `session_conflict` | another active job already exists on the same `thread_id` |
| `409` | `idempotency_conflict` | the same key points to another `thread_id` or user context |
| `503` | `job_enqueue_failed` | the request cannot be safely enqueued |


Notes:
- `X-User-Id` header as the source of user identity
- `idempotency_key` is optional (Duplicate request handling)
- `thread_id` must be a valid `UUID`
- single-`user_id` ownership per `thread_id`
- same `user_id + idempotency_key` reuses the existing job when the context matches

### 3.2 `GET /agent/api/jobs/{job_id}/events`
> Subscribe to SSE events

Request Header:
```http
X-User-Id: user-123
Last-Event-ID: 1743500000000-0
```

Response Event Payload:
```json
{
  "type": "messages",
  "ns": [],
  "data": [
    {
      "type": "AIMessageChunk",
      "data": {
        "content": "Hello. How can I help you?"
      }
    },
    {
      "langgraph_node": "model",
      "tags": ["seq:step:2"]
    }
  ],
  "event_id": "1743500000000-0"
}
```

Errors:
| Status | Code | Detail |
| --- | --- | --- |
| `401` | `missing_user_identity` | `X-User-Id` header is missing |
| `403` | `forbidden` | the job does not belong to the requesting user |
| `404` | `job_not_found` | the requested job does not exist |

Usage:
1. backend connects to `events_url`
2. backend consumes SSE events until completion or timeout
3. backend stores the latest SSE `event_id`
4. backend reconnects with `Last-Event-ID` when the stream is interrupted
5. backend stops reconnecting after the job reaches a terminal status

Notes:
- `Last-Event-ID` header supported for replay
- terminal status fallback when the final `complete` event is missing or delayed
- final drain attempt before terminal stream close
- maximum SSE connection duration controlled by `SSE_MAX_CONNECTION_SECONDS`

### 3.3 `GET /agent/api/jobs/{job_id}`
> Fetch the current job status

Request Header:
```http
X-User-Id: user-123
```

Response:
```json
{
  "job_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
  "status": "running",
  "thread_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
  "user_id": "user-123",
  "created_at": "2026-04-02T10:00:00+00:00",
  "started_at": "2026-04-02T10:00:01+00:00",
  "finished_at": null,
  "error": null,
  "metadata": null
}
```

Errors:
| Status | Code | Detail |
| --- | --- | --- |
| `401` | `missing_user_identity` | `X-User-Id` header is missing |
| `403` | `forbidden` | the job does not belong to the requesting user |
| `404` | `job_not_found` | the requested job does not exist |

### 3.4 `POST /agent/api/jobs/{job_id}/cancel`
> Request cancellation

Request Header:
```http
X-User-Id: user-123
```

Response:
```json
{
  "job_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
  "cancel_requested": true,
  "status": "running"
}
```

Errors:
| Status | Code | Detail |
| --- | --- | --- |
| `401` | `missing_user_identity` | `X-User-Id` header is missing |
| `403` | `forbidden` | the job does not belong to the requesting user |
| `404` | `job_not_found` | the requested job does not exist |

### 3.5 `GET /agent/api/ping`
> Liveness check

Response:
```json
{
  "status": true
}
```

### 3.6 `GET /agent/api/health`
> Readiness check

Response:
```json
{
  "status": true,
  "checks": {
    "runtime_config": true,
    "session_manager": true,
    "redis_client": true,
    "redis_ping": true
  }
}
```

Errors:
| Status | Code | Detail |
| --- | --- | --- |
| `503` | `service_unavailable` | one or more readiness checks failed |

---

## 4. Swagger
> Local API inspection and docs

- OpenAPI JSON: `/agent/openapi.json`
- Swagger UI: `/agent/swagger`
