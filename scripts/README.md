# Deployment Helper Scripts

These scripts turn Terraform outputs into repeatable deployment steps. Run them from the repository root after the first `environments/infra` Terraform apply.

## Prerequisites

- `terraform`
- `az`
- `jq`
- `uv`
- `docker`

Not every script needs every tool. Each script validates the commands it uses.

## `bootstrap-secrets.sh`

Sets all application secrets in Azure Key Vault.

```bash
scripts/bootstrap-secrets.sh --infra-dir environments/infra
```

It reads Terraform outputs, fetches Azure resource keys where possible, prompts for values that cannot be inferred, and writes the names defined in [`../environments/env/.env.keyvault`](../environments/env/.env.keyvault).

Preview without writing:

```bash
scripts/bootstrap-secrets.sh --infra-dir environments/infra --dry-run
```

Non-interactive example:

```bash
POSTGRES_ADMIN_PASSWORD="<postgres-password>" \
LANGFUSE_PUBLIC_KEY="<langfuse-public-key>" \
LANGFUSE_SECRET_KEY="<langfuse-secret-key>" \
REDIS_ACCESS_KEY="<redis-memory-access-key>" \
REDIS_STREAM_ACCESS_KEY="<redis-stream-access-key>" \
scripts/bootstrap-secrets.sh --infra-dir environments/infra --yes
```

Environment override convention:

```text
Key Vault secret: LANGFUSE-PUBLIC-KEY
Environment var:  LANGFUSE_PUBLIC_KEY
```

## `setup-azure-ai-search.sh`

Creates or updates the Azure AI Search index and uploads the sample documents from `examples/azure_ai_search`.

```bash
scripts/setup-azure-ai-search.sh --infra-dir environments/infra
```

Useful options:

```bash
scripts/setup-azure-ai-search.sh --infra-dir environments/infra --dry-run
scripts/setup-azure-ai-search.sh --infra-dir environments/infra --skip-documents
scripts/setup-azure-ai-search.sh --infra-dir environments/infra --index-name azure-agent-index
```

Environment overrides:

```env
AZURE_AI_SEARCH_ENDPOINT=
AZURE_AI_SEARCH_API_KEY=
AZURE_AI_SEARCH_INDEX_NAME=
AZURE_AI_SEARCH_API_VERSION=
```

## `upload-prompts.sh`

Uploads prompt files to the Terraform-created `prompts` Blob container.

```bash
scripts/upload-prompts.sh --infra-dir environments/infra
```

Useful options:

```bash
scripts/upload-prompts.sh --infra-dir environments/infra --dry-run
scripts/upload-prompts.sh --infra-dir environments/infra --prompt-dir src/azure_agent/prompts
```

Environment override:

```env
BLOB_CONNECTION_STRING=
```

## `build-push-images.sh`

Builds and pushes the API/worker/job image and the web image to ACR.

```bash
scripts/build-push-images.sh --infra-dir environments/infra
```

Useful options:

```bash
scripts/build-push-images.sh --infra-dir environments/infra --dry-run
scripts/build-push-images.sh --infra-dir environments/infra --platform linux/amd64
scripts/build-push-images.sh --infra-dir environments/infra --skip-web
scripts/build-push-images.sh --infra-dir environments/infra --skip-api-worker
```

Default image tags:

```hcl
api_worker_image_tag = "azure-agent:local"
web_image_tag        = "azure-agent-web:local"
```

Keep these tags aligned with `environments/infra/terraform.tfvars`.
