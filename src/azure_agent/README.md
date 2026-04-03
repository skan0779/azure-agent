<p align="center">
  <img src="/docs/icons/Azure-Agent.png" height="72" alt="Azure Agent" />
</p>

<h1 align="center">Azure Agent Package</h1>

<p align="center">
  Application package structure for the API, worker, graph, and runtime modules
</p>

---

## 1. Overview
> High-level guide to the `src/azure_agent` package

- `azure_agent` contains the runtime code for the API server, background worker, LangGraph agent, and supporting infrastructure
- The package is organized by runtime responsibility such as `api`, `graphs`, `session`, `jobs`, and `worker`
- Some folders have dedicated README files for deeper implementation details

---

## 2. Package Structure
> Main folders and their responsibilities

| Folder | Role | Notes |
| --- | --- | --- |
| `api/` | FastAPI application, routes, request/response schemas, and bundled Swagger assets | Exposes job APIs and SSE event streaming |
| `config/` | Runtime configuration models and environment-based config loading | Central place for TTLs, worker polling, and SSE timeout settings |
| `encoder/` | Bundled `tiktoken` cache files | Supports air-gapped or offline-friendly runtime environments |
| `graphs/` | LangGraph agent construction, runtime setup, and stream event shaping | Builds models, tools, memory, prompts, and graph execution flow |
| `infra/` | Shared infrastructure helpers for external services | Includes Azure Key Vault and Redis client creation |
| `jobs/` | Job queue and event persistence helpers on Redis Stream | Manages job creation, replay, patching, and result event append/read |
| `middlewares/` | Custom middleware used by the agent runtime | Currently used for custom stream events before agent/model execution |
| `prompts/` | Local YAML prompt files | Used as startup fallback when Blob prompt loading fails |
| `session/` | Per-`thread_id` session ownership, lock, and active job coordination | Prevents concurrent execution conflicts |
| `tools/` | External tool integrations exposed to the agent | Includes Azure AI Search based retrieval tooling |
| `worker/` | Background worker process that consumes queued jobs and runs the graph | Writes job status and streamed events back to Redis |
