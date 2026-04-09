<p align="center">
  <img src="/docs/icons/Azure-Agent.png" width="20%" alt="Azure Agent" />
</p>

<h1 align="center">Azure Agent</h1>

<p align="center">
  Production-ready Enterprise AI Chat Agent Template for Microsoft Azure
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-2563EB.svg" alt="License"></a>
  <a href="https://github.com/skan0779/azure-agent/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/skan0779/azure-agent?style=flat&color=1D4ED8" alt="GitHub Stars"></a>
  <a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-306998?logo=python&logoColor=white" alt="Python"></a>
  <img src="https://img.shields.io/badge/LangGraph-1F2937?logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/Redis-B91C1C?logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/PostgreSQL-1D4ED8?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/FastAPI-0F766E?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-2563EB?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/uv-7C3AED?logo=uv&logoColor=white" alt="uv">
</p>
<br>

## Quickstart
> Required and optional steps to deploy the services.

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
| Azure Key Vault | [Generate Secrets](./environments/env/README.md) |
| Log Analytics Workspace | - |

### 2. Create an Azure AI Search index
- Create the index schema
```bash
uv run python examples/azure_ai_search/create_index.py
```
- Upload index documents
```bash
uv run python examples/azure_ai_search/create_document.py
```
- [`README.md`](./examples/azure_ai_search/README.md)

### 3. Create an [Tavily](https://www.tavily.com/) account (optional)
- Create a Tavily API key

### 4. Upload prompt files to Azure Blob Storage (optional)
- Upload prompt files such as `example.yaml`
- Blob Storage is the primary source for prompts (fallback to the local prompt files)
- [`README.md`](./src/azure_agent/prompts/README.md)

### 5. Configure an Azure Key Vault Secrets
- Add secret values from [`.env.keyvault`](./environments/env/.env.keyvault)
- [`README.md`](./environments/env/README.md)

### 6. Build and Push the Docker Image
- Build the Docker image
- Run `az login`
- Push the Docker image to `Azure Container Registry`
- [`README.md`](./environments/deploy/README.md)

### 7. Deploy an Azure Container Apps
- Deploy `azure-agent-api` with ingress enabled on port `8080`
- Deploy `azure-agent-worker` with ingress disabled (set command override: `sh` & `-lc`, `uv run azure-agent-worker`)

### 8. Configure an Azure Container Apps
- Enable a Managed Identity
- Grant permissions to Managed Identity: `Key Vault Secrets User`, `Storage Blob Data Reader`, `ACR Pull`
- Set environment variables from [`.env.example`](./environments/env/.env.example)
- [`README.md`](./src/azure_agent/api/README.md)

### 9. Check Swagger & Status
- `https://<application-url>/agent/api/ping`
- `https://<application-url>/agent/api/health`
- `https://<application-url>/agent/swagger`

---

## Agent Feature & Checklist
> AI Agent Stack for Enterprise workloads on Microsoft Azure

| Category | Library | Resource |
| --- | --- | --- |
| Job Queue & Worker | [Redis Streams](https://redis.io/docs/latest/develop/data-types/streams) | Azure Managed Redis (OSS) |
| Session Management | [Session Manager(Custom)](./src/azure_agent/session/README.md) | Azure Managed Redis (OSS) |
| Model Routing | [model-router](https://ai.azure.com/catalog/models/model-router) | Azure AI Foundry |
| RAG | [AzureSearch](https://docs.langchain.com/oss/python/integrations/vectorstores/azuresearch) | Azure AI Search |
| Web Search | [TavilySearch](https://reference.langchain.com/python/langchain-tavily/tavily_search/TavilySearch) | - |
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
| Client Interface | - | - |

<!-- | Rate Limiting | ModelCallLimitMiddleware, ToolCallLimitMiddleware | - | -->

---

## Agent Architecture
> 

not yet

---

## License
> This project is licensed under the `Apache License 2.0`. See [LICENSE](./LICENSE) for details.

Bundled `Swagger UI assets` may require separate upstream license notice files during redistribution. If you update or re-bundle these assets, include all required third-party notices in the distributed package.
