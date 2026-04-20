<p align="center">
  <img src="/docs/icons/Docker.svg" height="72" alt="Docker" />
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
docker buildx build --platform linux/amd64 \
  -f environments/deploy/Dockerfile \
  -t azure-agent:local \
  --load .
```

---

## 3. Push Image to ACR
> Push Docker Image to Azure Container Registry
```bash
# Tag
docker tag azure-agent:local <acr-name>.azurecr.io/azure-agent:v1.x

# Push
docker push <acr-name>.azurecr.io/azure-agent:v1.x
```

---

## 4. Create Azure Container Apps
> Create separate Azure Container App resources for `API` and `Worker`. (Both resources use the same container image tag)

- `api` handles HTTP traffic.
- `worker` consumes Redis jobs and should not be exposed through ingress.

Recommended layout:
```text
ACA #1: azure-agent-api
- Image: <acr-name>.azurecr.io/azure-agent:v1.x
- Ingress: enabled
- Target port: 8080
- Command override: not required

ACA #2: azure-agent-worker
- Image: <acr-name>.azurecr.io/azure-agent:v1.x
- Ingress: disabled
- Command override: required
```

---

## 5. Setting Azure Container Apps (Command Override)
> The `Worker` Container App must override the default container command. (Application > Containers > Properties > Command override) Image default command starts the `API` server.

Worker override values:
```text
Command override:
sh

Arguments override:
-lc, uv run azure-agent-worker
```
