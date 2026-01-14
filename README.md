# <img src="docs/icons/Azure.svg" alt="Azure" height="28" /> Azure Agent
> Production-ready AI Agent Template optimized for Azure Cloud


---

## 2. Quick Start

### 2.1 Setup (uv)
> Use `uv` for dependency management.
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 2.2 Run Locally
> Start the FastAPI application
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 2.3 Run with Docker
> Build and run using the deployment compose file.
```bash
docker compose -f environments/deploy/docker-compose.yml up --build
```

### 2.4 Swagger API
> Swagger UI assets are bundled in `app/static` (load in air‑gapped environment)
```bash
http://<host>:8001/agent/swagger
```
