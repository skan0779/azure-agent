# <img src="/docs/icons/Azure-A.svg" height="28" /> Azure Agent
> Production-ready AI Agent Template optimized for Azure Cloud


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
> Start the FastAPI application (match the container module path).
```bash
PYTHONPATH=src/azure_agent uv run uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
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
