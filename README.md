<p align="center">
  <img src="/docs/icons/Azure-Agent.png" width="20%" alt="Azure Agent" />
</p>

<h1 align="center">Azure Agent</h1>

<p align="center">
  Production-ready AI Chatbot Agent Sample optimized for Microsoft Azure
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2563EB.svg" alt="License"></a>
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

### 1. Provision Azure resources
- Azure OpenAI: `gpt-4o-mini`, `text-embedding-3-large`
- Azure AI Foundry: `model-router` (set `Azure Content Safety`)
- Azure Managed Redis (OSS)
- Azure Managed Redis (Enterprise, required modules: `RedisJSON`, `RedisSearch`)
- Azure Database for PostgreSQL
- Azure Key Vault
- Azure Container Registry
- Azure Container Apps Environment
- Azure Container Apps: `azure-agent-api`, `azure-agent-worker`
- Azure Storage Account (Blob)
- Azure AI Search

### 2. Create an Azure AI Search index
- Create the index schema
- Upload index documents

### 3. Create an [`Tavily`](https://www.tavily.com/) account (optional)
- Create a Tavily API key

### 4. Upload prompt files to `Azure Blob Storage` (optional)
- Upload prompt files such as `example.yaml`
- Blob Storage is the primary source for prompts (fallback to the local prompt files)

### 5. Configure `Azure Key Vault` secrets
- Add secret values from [`.env.keyvault`](./environments/env/.env.keyvault)

### 6. Build and push the Docker image
- Build the Docker image
- Run `az login`
- Push the Docker image to `Azure Container Registry`

### 7. Deploy `Azure Container Apps`
- Deploy `azure-agent-api` with ingress enabled on port `8080`
- Deploy `azure-agent-worker` with ingress disabled (set command override: `sh` & `-lc`, `uv run azure-agent-worker`)

### 8. Configure `Azure Container Apps` Identities and Access
- Enable a `managed identity`
- Grant permissions to managed identity: `Key Vault Secrets User`, `Storage Blob Data Reader`
- Set environment variables from [`.env.example`](./environments/env/.env.example)

---

## Feature Checklist


## Resource Checklist


---

## License
This project is licensed under the MIT License. See `LICENSE` for details.
Bundled `Swagger UI assets` may require separate upstream license notice files during redistribution. If you update or re-bundle these assets, include all required third-party notices in the distributed package.
