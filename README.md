<p align="center">
  <img src="/docs/icons/Azure-Agent.png" height="72" alt="Azure Agent" />
</p>

<h1 align="center">Azure Agent</h1>

<p align="center">
  Production-ready AI Agent Sample optimized for Azure Cloud
</p>

---


## Project Structure



---


## Quick Start

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


## API Endpoints
- `GET /agent/api/ping`
- `POST /agent/api/user_query/stream` (SSE stream)
- `POST /agent/api/delete_thread`

## Swagger UI
> Swagger UI assets are bundled in `src/azure_agent/api/static` (air-gapped).
```bash
http://<your-azure-container-host>:8001/agent/swagger
```
