<!-- <p align="center">
  <img src="/docs/icons/Azure-Agent.png" width="20%" alt="Azure Agent" />
</p> -->
<p align="center">
  <img src="./docs/video/video.gif" width="100%" alt="Azure Agent demo" />
</p>

<h1 align="center">Azure Agent</h1>

<p align="center">
  Enterprise-Grade AI Chat Agent Template on Azure
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-2563EB.svg" alt="License"></a>
  <a href="https://github.com/skan0779/azure-agent/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/skan0779/azure-agent?style=flat&color=1D4ED8" alt="GitHub Stars"></a>
  <a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-306998?logo=python&logoColor=white" alt="Python"></a>
  <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/React-149ECA?logo=react&logoColor=white" alt="React">
  <img src="https://img.shields.io/badge/Next.js-111111?logo=nextdotjs&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/LangGraph-1F2937?logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/Redis-B91C1C?logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/PostgreSQL-1D4ED8?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/FastAPI-0F766E?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Fastify-000000?logo=fastify&logoColor=white" alt="Fastify">
  <img src="https://img.shields.io/badge/Docker-2563EB?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/PNPM-F69220?logo=pnpm&logoColor=white" alt="pnpm">
  <img src="https://img.shields.io/badge/uv-7C3AED?logo=uv&logoColor=white" alt="uv">
</p>
<br>

## Quickstart
> Required and optional steps to start the service.

### 1. Provision Azure Resources

| Resource | Notes |
| --- | --- |
| Azure OpenAI | Deploy `gpt-4o-mini`, `text-embedding-3-large` |
| Azure AI Foundry | Deploy `model-router` |
| Azure AI Search | [Create and Upload Document](./examples/azure_ai_search/README.md) |
| Azure AI Content Safety | - |
| Azure Managed Redis (Enterprise) | Required modules: `RedisJSON`, `RedisSearch` |
| Azure Managed Redis (OSS) | - |
| Azure Database for PostgreSQL | - |
| Azure Storage Account (Blob) | - |
| Azure Container Registry | - |
| Azure Container Apps Environment | - |
| Azure Container Apps | Deploy `azure-agent-api` service |
| Azure Container Apps | Deploy `azure-agent-worker` service |
| Azure Container Apps | Deploy `azure-agent-web` service |
| Azure Static Web Apps | Deploy `azure-agent-ui` service |
| Azure Container Apps Session Pool | Pool type: `Python` |
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

Build and push `azure-agent-api`, `azure-agent-worker` docker image:
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

### 6. Deploy and Configure an Azure Container Apps
> For more details, see [`README.md`](./environments/env/README.md).

Deploy `azure-agent-api`:
- Ingress : ✅
- Target port : 8080
- Security > Identity > System assigned: ✅
- Azure role assignments: `Key Vault Secrets User`, `ACR Pull`
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
- Application > Containers > Environment variables:
```env
KEY_VAULT_URL=<your-key-vault-url>
AGENT_API_BASE_URL=<your-azure-agent-api-url>
CORS_ORIGINS=https://<your-static-web-app-domain>
HOST=0.0.0.0
PORT=3001
```

### 7. Deploy and Configure an Azure Static Web Apps
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

### 8. Check Service Status (optional)
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

## Agent Feature & Checklist
> AI Agent Stack for Enterprise workloads on Microsoft Azure

| Category | Library | Resource |
| --- | --- | --- |
| Job Queue & Worker | [Redis Streams](https://redis.io/docs/latest/develop/data-types/streams) | Azure Managed Redis (OSS) |
| Session Management | [Session Manager(Custom)](./src/azure_agent/session/README.md) | Azure Managed Redis (OSS) |
| Model Routing | [model-router](https://ai.azure.com/catalog/models/model-router) | Azure AI Foundry |
| RAG | [AzureSearch](https://docs.langchain.com/oss/python/integrations/vectorstores/azuresearch) | Azure AI Search |
| Web Search | [Web Search (Grounding with Bing Search)](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/web-search) | - |
| Code Interpreter | [SessionsPythonREPLTool](https://learn.microsoft.com/en-us/azure/container-apps/sessions) | Azure Container Apps Session Pool |
| Sandbox (coding) | []() | Azure Container Apps Session Pool |
| Long-term Memory | [AsyncPostgresStore](https://docs.langchain.com/oss/python/langgraph/add-memory#example-using-postgres-store), [Langmem](https://github.com/langchain-ai/langmem) | Azure Database for PostgreSQL |
| Short-term Memory | [AsyncShallowRedisSaver](https://docs.langchain.com/oss/python/langgraph/add-memory#example-using-redis-checkpointer) | Azure Managed Redis (Enterprise) |
| Context Management | [SummarizationMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware) | Azure OpenAI |
| Moderation | [AzureContentModerationMiddleware](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain-middleware) | Azure AI Content Safety |
| Safety classifier | [AzurePromptShieldMiddleware](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain-middleware) | Azure AI Content Safety |
| PII filter | [PIIMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/pii/PIIMiddleware) | - |
| Prompt Management | [azure-storage-blob](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/storage/azure-storage-blob) | Azure Blob Storage |
| Secret Management  | [keyvault](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/keyvault) | Azure Key Vault |
| Streaming | [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming), [FastAPI SSE](https://fastapi.tiangolo.com/tutorial/server-sent-events) | Azure Managed Redis (OSS) |
| Observability | [Langfuse](https://github.com/langfuse/langfuse) | - |
| UI | [assistant-ui](https://github.com/assistant-ui/assistant-ui), [tool ui](https://www.tool-ui.com/) | Azure Static Web Apps  |
<!-- | Rate Limiting | ModelCallLimitMiddleware, ToolCallLimitMiddleware | - | -->

---

## Azure Resouece Architecture
> 

<p align="center">
  <img src="./docs/diagram/Azure-Resource-Architecture.png"
       alt="azure resource achitecture"
       style="width: 100%; height: auto;">
</p>


---

## License
> This project is licensed under the `Apache License 2.0`. See [LICENSE](./LICENSE) for details.

Bundled `Swagger UI assets` may require separate upstream license notice files during redistribution. If you update or re-bundle these assets, include all required third-party notices in the distributed package.
