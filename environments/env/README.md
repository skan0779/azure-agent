<h1 align="center">Environment & Key Vault</h1>

<p align="center">
  Define the secret values that must be stored in Azure Key Vault.
</p>

---

## 1. Key Vault Secrets
> Azure Key Vault stores secret values such as API keys, connection strings, and credentials.
> Container Apps should reference these secrets under **Security > Secrets**, then inject them into runtime environment variables with `secretref`.

Use [`./.env.keyvault`](./.env.keyvault) as the source of truth for Key Vault secret names.

```env
# Azure OpenAI
AZURE-OPENAI-ENDPOINT=
AZURE-OPENAI-API-KEY=
AZURE-OPENAI-API-VERSION=
AZURE-OPENAI-MAIN-MODEL=
AZURE-OPENAI-SMALL-MODEL=
AZURE-OPENAI-MAIN-MODEL-TIMEOUT=
AZURE-OPENAI-SMALL-MODEL-TIMEOUT=
AZURE-OPENAI-EMBEDDING-MODEL=
AZURE-OPENAI-EMBEDDING-DIMS=

# Azure AI Search
AZURE-AI-SEARCH-ENDPOINT=
AZURE-AI-SEARCH-API-KEY=
AZURE-AI-SEARCH-INDEX-NAME=
AZURE-AI-SEARCH-SEMANTIC-CONFIG=
AZURE-AI-SEARCH-API-VERSION=
AZURE-AI-SEARCH-TOP-K=

# Azure AI Content Safety
AZURE-AI-CONTENT-SAFETY-ENDPOINT=
AZURE-AI-CONTENT-SAFETY-API-KEY=

# Azure Container Apps Dynamic Sessions
AZURE-DYNAMIC-SESSIONS-PYTHON-POOL-ENDPOINT=
AZURE-DYNAMIC-SESSIONS-BASH-POOL-ENDPOINT=

# Azure Blob Storage
BLOB-CONNECTION-STRING=

# Azure Managed Redis (Enterprise)
REDIS-HOST=
REDIS-USERNAME=
REDIS-ACCESS-KEY=
REDIS-PORT=
REDIS-DB=

# Azure Managed Redis (OSS)
REDIS-STREAM-HOST=
REDIS-STREAM-USERNAME=
REDIS-STREAM-ACCESS-KEY=
REDIS-STREAM-PORT=

# Azure Database for Postgres (Worker)
POSTGRES-CONN-STRING=

# Azure Database for Postgres (Web)
POSTGRES-WEB-CONN-STRING=

# Langfuse
LANGFUSE-SECRET-KEY=
LANGFUSE-PUBLIC-KEY=
```

---

## 2. Set secrets from your machine

### 2.1 Ensure your account has one of these roles in Key Vault
> Key Vault Secrets Officer

### 2.2 Allow public access from specific IP addresses in Key Vault
> Add your Client IP address

### 2.3 Set secrets with the Azure CLI
```bash
az login
az keyvault secret set --vault-name <key-vault-name> --name SECRET_NAME --value "SECRET_VALUE"
```

---

## 3. When direct access is blocked (optional)
Use this path if you cannot reach the Key Vault due to network restrictions (no public access).

### 3.1 Assign a role
> Assign Key Vault Secrets Officer to your account.

### 3.2 Create a private endpoint and DNS
> Create and associate a private endpoint and DNS for the Key Vault.

### 3.3 Create a VM
> Create a VM in the same VNet/Subnet with a public IP and NSG inbound SSH rules.

### 3.4 Set secrets from the VM
> Virtual Machine > Connect > **Configure JIT + Request access**
```bash
# SSH into the VM
ssh -i "<local-key-path>" <vm-user-name>@<vm-public-ip>

# Restrict private key permissions
chmod 600 <local-key-path>

# Install Azure CLI (Debian/Ubuntu)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Azure login (device code flow)
az login --use-device-code

# Set a Key Vault secret
az keyvault secret set --vault-name <key-vault-name> --name SECRET_NAME --value "SECRET_VALUE"
```
