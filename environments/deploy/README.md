# Docker Deployment

This directory contains Docker assets for the Python runtime services:

- `azure-agent-api`
- `azure-agent-worker`
- `azure-agent-job`

The web service uses its own Dockerfile:

- [`../../apps/azure-agent-web/Dockerfile`](../../apps/azure-agent-web/Dockerfile)

## Recommended Flow

Build and push images from the repository root:

```bash
scripts/build-push-images.sh --infra-dir environments/infra
```

The script reads `acr_login_server` from Terraform output, logs in to ACR, and pushes:

```text
<acr-login-server>/azure-agent:local
<acr-login-server>/azure-agent-web:local
```

Preview without running Docker:

```bash
scripts/build-push-images.sh --infra-dir environments/infra --dry-run
```

For all script options, see [`../../scripts/README.md`](../../scripts/README.md).

## Image Tags

Keep image tags aligned with `environments/infra/terraform.tfvars`:

```hcl
api_worker_image_tag = "azure-agent:local"
web_image_tag        = "azure-agent-web:local"
```

Terraform adds the ACR login server when it creates Container Apps.

## Runtime Commands

The same Python image is used by `api`, `worker`, and `job`.

| Unit | Command |
| --- | --- |
| `azure-agent-api` | Dockerfile default `CMD` |
| `azure-agent-worker` | `sh -lc "uv run azure-agent-worker"` |
| `azure-agent-job` | `sh -lc "uv run --no-sync alembic upgrade head"` |

Container Apps and the Container App Job are created by Terraform after images and Key Vault secrets are ready:

```hcl
deploy_container_apps    = true
deploy_container_app_job = true
```

## Local Compose

[`docker-compose.yml`](./docker-compose.yml) is for local development only. It reads `../env/.env.dev` and starts a local Redis container.

The Azure deployment path does not use this Compose file. Azure runtime configuration is injected through Container Apps environment variables and Key Vault-backed secret references.
