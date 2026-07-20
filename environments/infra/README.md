# Azure Agent Infrastructure

Terraform template for the core Azure Agent infrastructure.

This stack is intentionally public-first for demos, development, and template validation. Production deployments should add private networking, stricter firewall rules, managed backups, monitoring, and policy controls.

## Scope

This Terraform stack creates:

- Resource Group
- Log Analytics Workspace
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

The optional demo Langfuse VM is managed by a separate Terraform stack:

- [`../langfuse-vm`](../langfuse-vm)

## Deployment Flow

Use two Terraform applies:

1. First apply creates shared Azure resources only.
2. Scripts configure secrets, search data, prompt files, and images.
3. Second apply creates Container Apps and the migration job.

This avoids first-apply failures before ACR images and Key Vault secret values exist.

## 1. First Apply

```bash
cd environments/infra
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Keep runtime services disabled for the first apply:

```hcl
deploy_container_apps    = false
deploy_container_app_job = false
```

Optionally create Azure OpenAI deployments if your subscription, region, and quota support the configured models:

```hcl
create_openai_deployments = true
```

Model deployment names default to:

- `gpt-5.4`
- `gpt-5.4-nano`
- `text-embedding-3-large`

## 2. Configure Runtime Dependencies

Run helper scripts from the repository root.

Set Key Vault application secrets:

```bash
scripts/bootstrap-secrets.sh --infra-dir environments/infra
```

Create the Azure AI Search index and upload sample documents:

```bash
scripts/setup-azure-ai-search.sh --infra-dir environments/infra
```

Upload prompt files to Blob Storage:

```bash
scripts/upload-prompts.sh --infra-dir environments/infra
```

Build and push Container App images to ACR:

```bash
scripts/build-push-images.sh --infra-dir environments/infra
```

For script options and dry-run examples, see [`../../scripts/README.md`](../../scripts/README.md).

## 3. Second Apply

After secrets and images are ready, update `terraform.tfvars`:

```hcl
deploy_container_apps    = true
deploy_container_app_job = true

worker_extra_env = {
  LANGFUSE_BASE_URL = "<terraform-output-langfuse-url>"
}

web_extra_env = {
  AZURE_AUTH_TENANT_ID      = "<your-directory-tenant-id>"
  AZURE_AUTH_API_CLIENT_ID  = "<your-api-app-registration-client-id>"
  AZURE_AUTH_REQUIRED_SCOPE = "access_as_user"
}
```

Then apply again:

```bash
terraform plan
terraform apply
```

Expected image tags:

```hcl
api_worker_image_tag = "azure-agent:local"
web_image_tag        = "azure-agent-web:local"
```

The ACR login server is added by Terraform when it creates the Container Apps.

## Key Vault and Secret References

Terraform creates the Key Vault and Container App secret references, but it does not write application secret values. This avoids storing secret values in Terraform state.

Secret naming follows this convention:

- Key Vault secret name: `AZURE-OPENAI-API-KEY`
- Container App secret key: `azure-openai-api-key`
- Runtime environment variable: `AZURE_OPENAI_API_KEY`

The expected Key Vault secret names are defined in [`../env/.env.keyvault`](../env/.env.keyvault) and exposed by the `key_vault_secret_names` output.

If `assign_current_user_key_vault_secrets_officer` is `true`, Terraform grants the current Azure principal permission to manage Key Vault secrets.

Some managed resources expose credentials to Terraform state. Protect the state file and prefer a remote backend for shared environments.

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

## Useful Outputs

```bash
terraform output
```

Commonly used outputs:

- `resource_group_name`
- `acr_login_server`
- `key_vault_uri`
- `storage_account_name`
- `search_endpoint`
- `openai_endpoint`
- `content_safety_endpoint`
- `redis_memory`
- `redis_stream`
- `postgres_connection_string_templates`
- `session_pool_endpoints`
- `container_app_urls`
- `static_web_app_url`
- `static_web_app_api_key`
- `key_vault_secret_names`
- `resource_names`

## Naming

Most resources use the `azure-agent-dev-<suffix>-*` pattern:

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
