<p align="center">
  <img src="/docs/icons/Docker.svg.svg" height="72" alt="Docker" />
</p>

<h1 align="center">Deployment (Docker)</h1>

<p align="center">
  Build Docker Image and Push Image to <strong>Azure Container Registry</strong>
</p>

---

## 1. Azure Login
> Create and login to Azure resources
```bash
# Azure Login and Select Subscription
az login

# Azure Container Registry Login
az acr login -n <acr-name>
```

---

## 2. Build Image
> Build docker image with Docker Desktop
```bash
# Build Image
docker build -f environments/deploy/Dockerfile -t azure-agent:local .

# Run (Optional)
docker run --rm -p 8080:8080 -e PORT=8080 \
  --env-file environments/env/.env.dev \
  azure-agent:local
```

## 3. Push Image to ACR
> Push Docker Image to Azure Container Registry
```bash
# Tag
docker tag azure-agent:local <acr-name>.azurecr.io/azure-agent:v1.x

# Push
docker push <acr-name>.azurecr.io/azure-agent:v1.x
```

