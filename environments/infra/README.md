# Azure Agent Infrastructure

Terraform template for a public-first Azure Agent deployment.

This template intentionally starts without VNet, private endpoints, private DNS, or VM jump hosts. The first goal is a low-friction deployment path for development, demos, and template validation. Production deployments should add private networking and stricter firewall rules.

## Scope

The current implementation creates:

- Resource Group
- Log Analytics Workspace
- Shared naming convention for the remaining Azure Agent resources
- Azure Container Registry
- Key Vault
- Storage Account
- Blob containers: `prompts`, `files`
- Azure AI Search
- Azure OpenAI account
- Optional Azure OpenAI model deployments
- Azure AI Content Safety
- Azure Managed Redis for memory/checkpoint features
- Azure Managed Redis for streams/sessions/SSE replay
- PostgreSQL Flexible Server
- PostgreSQL databases for agent and web
- Container Apps Environment
- User-assigned managed identities for `api`, `worker`, `web`, and `job`
- Optional Container Apps for `api`, `worker`, and `web`
- Optional manual Container App Job
- Container Apps Session Pools for Python and Bash
- Static Web Apps
- Managed identity role assignments

## Network Model

Default mode is public-first:

- `api` Container App: external ingress, target port `8080`
- `web` Container App: external ingress, target port `3001`
- `worker` Container App: ingress disabled
- VNet/private endpoints: not created
- Azure service public endpoints: enabled

`allowed_ip_ranges` is reserved for restricted-public mode. When you enable IP restrictions later, include both your client IPs and the Container Apps outbound IPs, otherwise the apps may deploy but fail at runtime.

PostgreSQL uses a quickstart firewall rule by default:

```hcl
postgres_allow_public_access_from_all_ips = true
```

Disable this for restricted deployments and set `postgres_firewall_rules` explicitly.

## Key Vault

The application secret names are defined in [`../env/.env.keyvault`](../env/.env.keyvault). This Terraform template exposes the expected names as `key_vault_secret_names`.

Terraform creates the Key Vault resource, but does not write application secrets. This avoids storing secret values in Terraform state.

After `terraform apply`, set values manually. For example:

```bash
az keyvault secret set \
  --vault-name "<key-vault-name>" \
  --name BLOB-CONNECTION-STRING \
  --value "<storage-account-connection-string>"

az keyvault secret set \
  --vault-name "<key-vault-name>" \
  --name AZURE-OPENAI-ENDPOINT \
  --value "<openai-endpoint>"

az keyvault secret set \
  --vault-name "<key-vault-name>" \
  --name AZURE-AI-SEARCH-ENDPOINT \
  --value "<search-endpoint>"

az keyvault secret set \
  --vault-name "<key-vault-name>" \
  --name POSTGRES-CONN-STRING \
  --value "postgresql://<user>:<password>@<postgres-fqdn>:5432/azure_agent?sslmode=require"

az keyvault secret set \
  --vault-name "<key-vault-name>" \
  --name POSTGRES-WEB-CONN-STRING \
  --value "postgresql://<user>:<password>@<postgres-fqdn>:5432/azure_agent_web?sslmode=require"
```

If `assign_current_user_key_vault_secrets_officer` is `true`, Terraform grants the current Azure principal permission to manage Key Vault secrets.

Terraform does not write secrets to Key Vault, but some managed resources expose credentials to Terraform state. Protect the state file and prefer a remote backend for shared environments.

Set the session pool endpoints from Terraform outputs:

```bash
az keyvault secret set \
  --vault-name "<key-vault-name>" \
  --name AZURE-DYNAMIC-SESSIONS-PYTHON-POOL-ENDPOINT \
  --value "<python-session-pool-endpoint>"

az keyvault secret set \
  --vault-name "<key-vault-name>" \
  --name AZURE-DYNAMIC-SESSIONS-BASH-POOL-ENDPOINT \
  --value "<bash-session-pool-endpoint>"
```

## Azure OpenAI Deployments

Model deployments are disabled by default because model availability and quota vary by subscription and region.

To create deployments, set:

```hcl
create_openai_deployments = true
```

Default deployment names:

- `gpt-5.4`
- `gpt-5.4-nano`
- `text-embedding-3-large`

Use the deployment names as these Key Vault secret values:

- `AZURE-OPENAI-MAIN-MODEL`
- `AZURE-OPENAI-SMALL-MODEL`
- `AZURE-OPENAI-EMBEDDING-MODEL`

## Container Apps

Container Apps are disabled by default:

```hcl
deploy_container_apps = false
```

This avoids a first `terraform apply` failure before container images exist in ACR. Recommended flow:

1. Run `terraform apply` with `deploy_container_apps = false`.
2. Push images to the output `acr_login_server`.
3. Set `deploy_container_apps = true`.
4. Run `terraform apply` again.

Expected image tags:

```hcl
api_worker_image_tag = "azure-agent:local"
web_image_tag        = "azure-agent-web:local"
```

The API and web apps have external ingress. The worker has no ingress.

Static Web Apps is created by Terraform, but the UI still needs a separate build/deploy pipeline. Use `static_web_app_api_key` as the GitHub Actions secret `AZURE_STATIC_WEB_APPS_API_TOKEN` and set `NEXT_PUBLIC_AGENT_WEB_URL` to the `web` Container App URL.

## Quickstart

```bash
cd environments/infra
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
```

Apply only after reviewing the plan:

```bash
terraform apply
```

## Naming

Most resources use the `azure-agent-*` pattern:

```text
azure-agent-dev-<suffix>-api
azure-agent-dev-<suffix>-worker
azure-agent-dev-<suffix>-web
```

Azure globally unique resources with stricter naming rules use compact names without hyphens:

```text
azureagentdev<suffix>acr
azureagentdev<suffix>st
```

Set `name_suffix` in `terraform.tfvars` when you want predictable names. If it is empty, Terraform generates a stable random suffix and stores it in state.
