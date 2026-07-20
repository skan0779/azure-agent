# UI and Web Gateway

This directory contains the frontend-facing services:

- `azure-agent-ui`: Next.js chat UI deployed to Azure Static Web Apps.
- `azure-agent-web`: Fastify web gateway deployed to Azure Container Apps.

The recommended deployment flow is defined in the root [Quickstart](../README.md#quickstart). Terraform creates the Azure Static Web Apps resource, the `azure-agent-web` Container App, and the required Key Vault-backed secret references.

## Web Gateway

Build and push the web image with the repository helper script:

```bash
scripts/build-push-images.sh --infra-dir environments/infra
```

Terraform deploys `azure-agent-web` during the second infrastructure apply:

```hcl
deploy_container_apps = true

web_extra_env = {
  AZURE_AUTH_TENANT_ID      = "<your-directory-tenant-id>"
  AZURE_AUTH_API_CLIENT_ID  = "<your-api-app-registration-client-id>"
  AZURE_AUTH_REQUIRED_SCOPE = "access_as_user"
}
```

Runtime configuration is injected through Container Apps environment variables and Key Vault-backed secret references. `POSTGRES_WEB_CONN_STRING` is provided from the `POSTGRES-WEB-CONN-STRING` Key Vault secret.

## Static Web App

Set this GitHub Actions secret:

```env
AZURE_STATIC_WEB_APPS_API_TOKEN=<azure-agent-ui-deployment-token>
```

You can get the deployment token from Terraform:

```bash
terraform -chdir=environments/infra output -raw static_web_app_api_key
```

Set these GitHub Actions variables:

```env
NEXT_PUBLIC_AGENT_WEB_URL=<azure-agent-web-application-url>
NEXT_PUBLIC_AZURE_TENANT_ID=<your-directory-tenant-id>
NEXT_PUBLIC_AZURE_CLIENT_ID=<your-ui-app-registration-client-id>
NEXT_PUBLIC_AZURE_API_SCOPE=api://<your-api-app-registration-client-id>/access_as_user
```

You can get the web URL after the runtime Terraform apply:

```bash
terraform -chdir=environments/infra output -json container_app_urls | jq -r .web
```

Deploy `azure-agent-ui` with the GitHub Actions workflow:

```bash
git push origin main
```
