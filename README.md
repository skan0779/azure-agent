<p align="center">
  <img src="/docs/icons/Azure-Agent.png" width="20%" alt="Azure Agent" />
</p>

<h1 align="center">Azure Agent</h1>

<p align="center">
  Production-ready AI Agent Sample optimized for Azure
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/skan0779/azure-agent/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/skan0779/azure-agent?style=flat" alt="GitHub Stars"></a>
  <a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <img src="https://img.shields.io/badge/Microsoft_Azure-0078D4?logo=microsoftazure&logoColor=white" alt="Microsoft Azure">
  <img src="https://img.shields.io/badge/LangGraph-121212?logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white" alt="uv">
</p>

<br>


## Quickstart

### Setup (uv)
> Use `uv` for dependency management.
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### Environment (required)
> `KEY_VAULT_URL` is required; secrets are pulled from Azure Key Vault.
> Check `environments/env/README.md` for the secret value list.
```bash
export KEY_VAULT_URL="<your-azure-key-vault-url>"
```

### Run Locally (optional)
> Start the FastAPI application via the installed package entrypoint.
```bash
# Re-sync after script changes
uv sync

# API
uv run azure-agent-api
```

> Start the background worker.
```bash
uv run azure-agent-worker
```

### Run with Docker (optional)
> Build and run using the deployment compose file.
```bash
docker compose -f environments/deploy/docker-compose.yml up --build
```


---

## Swagger UI
> Swagger UI assets are bundled in `src/azure_agent/api/static` (air-gapped).
```bash
http://<your-azure-container-host>:8001/agent/swagger
```

---

## License
This project is licensed under the MIT License.
See `LICENSE` for details.

## Third-Party Notices
Bundled Swagger UI assets may require separate upstream license notice files during redistribution.
If you update or re-bundle these assets, include all required third-party notices in the distributed package.
