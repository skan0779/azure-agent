<h1 align="center">Azure Agent</h1>

<p align="center">
  Enterprise-Grade AI Chat Agent Template on Azure
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-2563EB.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Docker-2563EB?logo=docker&logoColor=white" alt="Docker">
</p>

<p align="center">
  <img src="./docs/video/video.gif" width="100%" alt="Azure Agent demo" />
</p>

## Architecture

> [!NOTE]
> This service consists of five deployment units: `azure-agent-ui`, `azure-agent-web`, `azure-agent-api`, `azure-agent-worker`, and `azure-agent-job`. The `ui` and `web` units provide the frontend-facing layer, while the `api` and `worker` units run the core AI agent runtime. `job` unit used for database migrations via Alembic.

<p align="center">
  <img src="./docs/diagram/Azure-Resource-Architecture.png"
       alt="azure resource achitecture"
       style="width: 100%; height: auto;">
</p>

---

## Quickstart
> Required and optional steps to start the service.

### 1. Provision Azure Resources

| Resource | Notes |
| --- | --- |
| Azure OpenAI | Deploy `gpt-5.4-nano`, `text-embedding-3-large` |
| Azure AI Foundry | Deploy `gpt-5.4` |
| Azure AI Search | [Create and Upload Document](./examples/azure_ai_search/README.md) |
| Azure AI Content Safety | - |
| Azure Managed Redis (Enterprise) | Required modules: `RedisJSON`, `RedisSearch` |
| Azure Managed Redis (OSS) | - |
| Azure Database for PostgreSQL | - |
| Azure Storage Account (Blob) | Create `files`, `prompts` container |
| Azure Container Registry | - |
| Azure Container Apps Environment | - |
| Azure Container Apps | Deploy `azure-agent-api` service |
| Azure Container Apps | Deploy `azure-agent-worker` service |
| Azure Container Apps | Deploy `azure-agent-web` service |
| Azure Static Web Apps | Deploy `azure-agent-ui` service |
| Azure Container App Job | Trigger type: `Manual`, Run `azure-agent-job` |
| Azure Container Apps Session Pool | Pool type: `Python` |
| Azure Container Apps Session Pool | Pool type: `Shell` |
| Azure Key Vault | [Generate Secrets](./environments/env/README.md) |
| Log Analytics Workspace | - |

### 2. Create an Azure AI Search index
> This repository provides an example Azure AI Search configuration using Microsoft Learn documentation as a sample RAG data source. In practice, adapt your own data. For more details, see [`README.md`](./examples/azure_ai_search/README.md).

Create index schema and Upload index documents
```bash
uv run python examples/azure_ai_search/create_index.py

uv run python examples/azure_ai_search/create_document.py
```

### 3. Upload prompt files to Azure Blob Storage (optional)
> This repository provides an example system prompt, [`example.yaml`](./src/azure_agent/prompts/example.yaml). If no prompt files are found in Azure Blob Storage, the application uses the local `example.yaml` file as a fallback. For production use, replace it with your own system prompt. For more details, see [`README.md`](./src/azure_agent/prompts/README.md).

Upload prompt file to blob
```bash
az storage blob upload \
  --connection-string "<your-blob-connection-string>" \
  --container-name "<your-blob-container-name>" \
  --file src/azure_agent/prompts/example.yaml \
  --name example.yaml \
  --overwrite
```

### 4. Configure an Azure Key Vault Secrets
> Store the values defined in [`.env.keyvault`](./environments/env/.env.keyvault) as secrets in Azure Key Vault. For more details, see [`README.md`](./environments/env/README.md).


Set the Key Vault Secrets:
```bash
az login

az keyvault secret set \
  --vault-name "<your-key-vault-name>" \
  --name "<secret-name>" \
  --value "<secret-value>"
```

### 5. Build and Push Docker Image to Azure Container Registry
> For more details, see [`README.md`](./environments/deploy/README.md) and [`README.md`](./apps/README.md).

Build and push `azure-agent-api`, `azure-agent-worker`, `azure-agent-job` docker image:
```bash
az login

az acr login -n "<your-acr-name>"

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -f environments/deploy/Dockerfile \
  -t "<your-acr-name>".azurecr.io/azure-agent:local \
  --push .
```

Build and push `azure-agent-web` docker image:
```bash
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -f apps/azure-agent-web/Dockerfile \
  -t "<your-acr-name>".azurecr.io/azure-agent-web:local \
  --push .
```

### 6. Run Azure Container App Job

Add a container `azure-agent-job`:
- Image source: Azure Conatiner Registry
- Managed identity: System assigned Identity (environment)
- Command override: `sh`
- Arguments override: `-lc, uv run --no-sync alembic upgrade head`
- Application > Containers > Environment variables:
```env
KEY_VAULT_URL=<your-key-vault-url>
```
- Settings > Identity > System assigned: ✅
- Settings > Identity > Azure role assignments: `Key Vault Secrets User`, `ACR Pull`

### 7. Deploy and Configure an Azure Container Apps
> For more details, see [`README.md`](./environments/env/README.md).

Deploy `azure-agent-api`:
- Ingress : ✅
- Target port : 8080
- Security > Identity > System assigned: ✅
- Azure role assignments: `Key Vault Secrets User`, `ACR Pull`
- Application > Scale > Min replicas: 1
- Application > Containers > Environment variables:
```env
KEY_VAULT_URL=<your-key-vault-url>
SSE_MAX_CONNECTION_SECONDS=600 # 10 minutes
JOB_TTL_SECONDS=86400 # 1 day
EVENT_TTL_SECONDS=86400 # 1 day
IDEMPOTENCY_TTL_SECONDS=86400 # 1 day
SESSION_TTL_SECONDS=3600 # 1 hour
SESSION_RESERVATION_TTL_SECONDS=300 # 5 minutes
SESSION_LOCK_TTL_SECONDS=90 # 1.5 minutes
```

Deploy `azure-agent-worker`:
- Ingress : ❎
- Command override: `sh`
- Arguments override: `-lc, uv run azure-agent-worker`
- Security > Identity > System assigned: ✅
- Azure role assignments: `Key Vault Secrets User`, `Storage Blob Data Reader`, `ACR Pull`, `Azure ContainerApps Session Executor`
- Application > Scale > Min replicas: 1
- Application > Containers > Environment variables:
```env
KEY_VAULT_URL=<your-key-vault-url>
JOB_TTL_SECONDS=86400 # 1 day
EVENT_TTL_SECONDS=86400 # 1 day
IDEMPOTENCY_TTL_SECONDS=86400 # 1 day
SESSION_TTL_SECONDS=3600 # 1 hour
SESSION_RESERVATION_TTL_SECONDS=300 # 5 minutes
SESSION_LOCK_TTL_SECONDS=90 # 1.5 minutes
WORKER_HEARTBEAT_INTERVAL_SECONDS=15 # 15 seconds
WORKER_PENDING_CLAIM_IDLE_MS=300000 # 5 minutes
WORKER_PENDING_CLAIM_COUNT=2 # 2 entries per reclaim cycle
WORKER_READ_BLOCK_MS=10000 # 10 seconds
WORKER_READ_COUNT=1 # 1 entry per read
```

Deploy `azure-agent-web`:
- Ingress : ✅
- Target port : 3001
- Security > Identity > System assigned: ✅
- Azure role assignments: `Key Vault Secrets User`, `ACR Pull`
- Application > Scale > Min replicas: 1
- Application > Containers > Environment variables:
```env
KEY_VAULT_URL=<your-key-vault-url>
AGENT_API_BASE_URL=<your-azure-agent-api-url>
CORS_ORIGINS=https://<your-static-web-app-domain>
HOST=0.0.0.0
PORT=3001
```

### 8. Deploy and Configure an Azure Static Web Apps
> For more details, see [`README.md`](./apps/README.md).

Deploy `azure-agent-ui`:
- Source: Other
- Deployment authorization policy: Deployment Token

Setting Github Actions (Github Repository > Settings > Secrets and variables > Actions):
- New repository secret: `AZURE_STATIC_WEB_APPS_API_TOKEN`: `azure-agent-ui` Deployment Token
- New repository variables: `NEXT_PUBLIC_AGENT_WEB_URL`: `azure-agent-web` Application Url

Deploy `azure-agent-ui` via github workflow:
```bash
git push origin main
```

### 9. Check Service Status (optional)
> For more details, see [`README.md`](./src/azure_agent/api/README.md).

Check `azure-agent-api` status:
- `https://<azure-agent-api-url>/agent/api/ping`
- `https://<azure-agent-api-url>/agent/api/health`
- `https://<azure-agent-api-url>/agent/swagger`

Check `azure-agent-web` status:
- `https://<azure-agent-web-url>/health`

Check `azure-agent-ui` status:
- `https://<azure-agent-ui-url>`

---

## Agent Features
> AI Agent Stack for Enterprise workloads on Microsoft Azure

| Category | Library | Resource |
| --- | --- | --- |
| RAG | [AzureSearch](https://docs.langchain.com/oss/python/integrations/vectorstores/azuresearch) | Azure AI Search |
| Web Search | [Web search](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/web-search) | Azure OpenAI |
| Sandbox Environment | [SessionsBashBackend](https://learn.microsoft.com/en-us/azure/container-apps/sessions) | Azure Container Apps Session Pool (shell) |
| Code Interpreter | [SessionsPythonREPLTool](https://learn.microsoft.com/en-us/azure/container-apps/sessions) | Azure Container Apps Session Pool (python) |
| Short-term Memory | [AsyncShallowRedisSaver](https://docs.langchain.com/oss/python/langgraph/add-memory#example-using-redis-checkpointer) | Azure Managed Redis (Enterprise) |
| Long-term Memory | [AsyncPostgresStore](https://docs.langchain.com/oss/python/langgraph/add-memory#example-using-postgres-store), [MemoryMiddleware](https://docs.langchain.com/oss/python/deepagents/memory) | Azure Database for PostgreSQL |
| Skiils | [SkillsMiddleware](https://docs.langchain.com/oss/python/deepagents/skills) | Azure Database for PostgreSQL |
| Context Management | [SummarizationMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware) | Azure OpenAI |
| Agent Orchestration | [SubAgentMiddleware](https://reference.langchain.com/python/deepagents/middleware/subagents/SubAgentMiddleware) | - |
| Task Management | [TodoListMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/todo/TodoListMiddleware) | Azure Managed Redis (Enterprise) |
| File Management | [FilesystemMiddleware](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware) | Azure Database for PostgreSQL |
| Rate Limiting | [ModelCallLimitMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/model_call_limit/ModelCallLimitMiddleware), [ToolCallLimitMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/tool_call_limit/ToolCallLimitMiddleware), .. | - |
| Content Moderation | [AzureContentModerationMiddleware](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain-middleware) | Azure AI Content Safety |
| Prompt Sheild | [AzurePromptShieldMiddleware](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain-middleware) | Azure AI Content Safety |
| PII | [PIIMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/pii/PIIMiddleware) | - |
| Session Management | [Session Manager(Custom)](./src/azure_agent/session/README.md) | Azure Managed Redis (OSS) |
| Stream Management(Job/Worker) | [Redis Streams](https://redis.io/docs/latest/develop/data-types/streams) | Azure Managed Redis (OSS) |
| Real-time interaction | [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming), [SSE](https://fastapi.tiangolo.com/tutorial/server-sent-events) | Azure Managed Redis (OSS) |
| Prompt Management | [azure-storage-blob](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/storage/azure-storage-blob) | Azure Blob Storage |
| Secret Management  | [keyvault](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/keyvault) | Azure Key Vault |
| Observability | [Langfuse](https://github.com/langfuse/langfuse) | - |
| UI | [assistant-ui](https://github.com/assistant-ui/assistant-ui), [tool ui](https://www.tool-ui.com/) | Azure Static Web Apps  |

---

## License
> This project is licensed under the [Apache License 2.0](./LICENSE).

Bundled `Swagger UI assets`, `OpenAI Skills` may require separate upstream license notice files during redistribution. If you update or re-bundle these assets, include all required third-party notices in the distributed package.