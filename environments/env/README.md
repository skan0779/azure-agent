<p align="center">
  <img src="/docs/icons/Key-Vault.svg" height="72" alt="Key Vault" />
</p>

<h1 align="center">Environment (Key Vault)</h1>

<p align="center">
  Define secret values in Azure Key Vault based on <code>.env.example</code>. Provide <code>KEY_VAULT_URL</code> as an environment variable when creating the container.
</p>

---

## 1. Direct access (recommended)

### 1.1 Ensure your account has one of these roles in Key Vault
> Key Vault Secrets Officer, Key Vault Secrets User

### 1.2 Allow public access from specific IP addresses in Key Vault
> Add your Client IP address

### 1.3 Set secrets with the Azure CLI
```bash
az login
az keyvault secret set --vault-name <key-vault-name> --name SECRET_NAME --value "SECRET_VALUE"
```

---

## 2. When direct access is blocked (optional)
Use this path if you cannot reach the Key Vault due to network restrictions (no public access).

### 2.1 Assign a role
> Assign Key Vault Secrets Officer or Key Vault Secrets User to your account.

### 2.2 Create a private endpoint and DNS
> Create and associate a private endpoint and DNS for the Key Vault.

### 2.3 Create a VM
> Create a VM in the same VNet/Subnet with a public IP and NSG inbound SSH rules.

### 2.4 Set secrets from the VM
```bash
# SSH into the VM
ssh -i "<local-key-path>" <vm-user>@<vm-public-ip>

# Restrict private key permissions
chmod 600 <local-key-path>

# Install Azure CLI (Debian/Ubuntu)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Azure login (device code flow)
az login --use-device-code

# Set a Key Vault secret
az keyvault secret set --vault-name <key-vault-name> --name SECRET_NAME --value "SECRET_VALUE"
```
