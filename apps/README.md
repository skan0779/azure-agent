<h1 align="center">UI</h1>

<p align="center">
  UI for azure-agent (frontend UI & backend server)
</p>
<br>


## Qucikstart
> deploy `azure-agent-web`(backend) and `azure-agent-ui`(frontend) for website & mobile UI

### 1. Create Azure Static Web App
> create `azure-agent-ui` azure resource

Build Details:
- Source: Other
- Deployment authorization policy: Deployment Token

### 2. Build and Push Docker Image
> build docker image and push image to Azure Container Registry

```bash
az login --use-device-code

az acr login -n <your-acr-name>

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -f apps/azure-agent-web/Dockerfile \
  -t <your-acr-name>.azurecr.io/azure-agent-web:v1 \
  --push .
```

### 3. Setting Key Vault Secrets
> add `POSTGRES-WEB-CONN-STRING` secrets

```bash
az keyvault secret set --vault-name <your-key-vault-name> --name POSTGRES-WEB-CONN-STRING --value <your-postgres-connection-string>
```

### 4. Create Azure Container App
> deploy `azure-agent-web` backend server (gateway server for `azure-agent-ui` and `azure-agent-api`)

Network:
- Ingress : ✅
- Target port : 3001

Security:
- Security > Identity > System assigned: ✅
- Azure role assignments: Key Vault Secrets User, ACR Pull

Application:
- Application > Containers > Environment variables:
```env
KEY_VAULT_URL=<your-key-vault-url>
AGENT_API_BASE_URL=<your-azure-agent-api-url>
CORS_ORIGINS=https://<your-static-web-app-domain>
AZURE_AUTH_TENANT_ID=<directory-tenant-id>
AZURE_AUTH_API_CLIENT_ID=<azure-agent-web-api-application-client-id>
AZURE_AUTH_REQUIRED_SCOPE=access_as_user
HOST=0.0.0.0
PORT=3001
```

### 5. Setting Github Actions
> Github Repository > Settings > Secrets and variables > Actions

New repository secret:
`AZURE_STATIC_WEB_APPS_API_TOKEN`: Azure Static Web App Deployment Token

New repository variables:
`NEXT_PUBLIC_AGENT_WEB_URL`: Azure Container App Application Url (azure-agent-web)
`NEXT_PUBLIC_AZURE_TENANT_ID`: Directory tenant ID
`NEXT_PUBLIC_AZURE_CLIENT_ID`: azure-agent-ui Application client ID
`NEXT_PUBLIC_AZURE_API_SCOPE`: `api://<azure-agent-web-api-application-client-id>/access_as_user`

### 6. Deploy UI via Github Actions
> deploy `azure-agent-ui` via github workflow

```bash
git push origin main
```
