

### 1. Create Static Web App

Build Details:
Source = Other
Deployment authorization policy: Deployment Token

### 3. Create and Deploy Azure Container App

```bash
docker build -f apps/azure-agent-web/Dockerfile -t <acr-name>.azurecr.io/azure-agent-web:latest .
docker push <acr-name>.azurecr.io/azure-agent-web:latest
```

Environment Variables:
AGENT_API_BASE_URL=http://<your-azure-agent-api-url>
CORS_ORIGINS=https://<your-swa-domain>
DEFAULT_USER_ID=dev-user
HOST=0.0.0.0
PORT=3001

### 2. Setting Github Repository Actions

Copy Deployment token:
Azure Portal -> Static Web App -> Manage deployment token

Add New Repository secret:
Settings -> Secrets and Variables -> Actions -> Secrets -> AZURE_STATIC_WEB_APPS_API_TOKEN: SWA deployment token

Add New Repository variables:
Settings -> Secrets and Variables -> Actions -> Variables -> NEXT_PUBLIC_AGENT_WEB_URL: azure-agent-web application url