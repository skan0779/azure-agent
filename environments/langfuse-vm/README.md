# Langfuse VM Infrastructure

Optional Terraform stack for a demo/development Langfuse VM.

This stack creates only Azure VM infrastructure:

- Virtual Network and subnet
- Network Security Group
- Public IP
- Network Interface
- Ubuntu Linux VM

Install Docker and run Langfuse over SSH after Terraform completes. For enterprise production use, run Langfuse on Kubernetes with managed dependencies, backups, monitoring, and network controls.

## Quickstart

Run the core infrastructure stack first and get its resource group name:

```bash
cd environments/infra
terraform output -raw resource_group_name
```

Then deploy the Langfuse VM:

```bash
cd ../langfuse-vm
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Use the outputs after apply:

```bash
terraform output -raw ssh_command
terraform output -raw langfuse_url
```

SSH into the VM and install Docker:

```bash
ssh azureuser@<vm-public-ip>

sudo apt-get update
sudo apt-get install -y ca-certificates curl git

sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker
```

Deploy Langfuse:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Generate values for Langfuse deployment secrets.
openssl rand -base64 32
openssl rand -hex 32

docker compose up -d
docker compose ps
```

Open Langfuse:

```txt
http://<vm-public-ip>:3000
```

Create an organization and project, then create a new API key from the project settings.

Store the generated API keys in Azure Key Vault:

```env
LANGFUSE-PUBLIC-KEY=<your-langfuse-public-key>
LANGFUSE-SECRET-KEY=<your-langfuse-secret-key>
```

Use the Terraform output as the worker plain environment variable:

```env
LANGFUSE_BASE_URL=<terraform-output-langfuse-url>
```
