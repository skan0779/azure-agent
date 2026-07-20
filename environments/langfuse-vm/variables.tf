variable "subscription_id" {
  description = "Azure subscription ID. Leave null to use ARM_SUBSCRIPTION_ID or the active Azure CLI subscription."
  type        = string
  default     = null
}

variable "resource_group_name" {
  description = "Existing resource group name from the core Azure Agent Terraform stack."
  type        = string
}

variable "project_name" {
  description = "Base project name used for Azure resource naming."
  type        = string
  default     = "azure-agent"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,30}[a-z0-9]$", var.project_name))
    error_message = "project_name must use lowercase letters, numbers, and hyphens, start with a letter, and end with a letter or number."
  }
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "dev"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,12}[a-z0-9]$", var.environment))
    error_message = "environment must use lowercase letters, numbers, and hyphens, start with a letter, and end with a letter or number."
  }
}

variable "name_suffix" {
  description = "Optional lowercase suffix to align names with the core Azure Agent Terraform stack."
  type        = string
  default     = ""

  validation {
    condition     = var.name_suffix == "" || can(regex("^[a-z0-9]{3,10}$", var.name_suffix))
    error_message = "name_suffix must be empty or 3-10 lowercase letters/numbers."
  }
}

variable "tags" {
  description = "Tags applied to Azure resources."
  type        = map(string)
  default     = {}
}

variable "admin_username" {
  description = "Linux administrator username for the Langfuse VM."
  type        = string
  default     = "azureuser"
}

variable "ssh_public_key" {
  description = "SSH public key content. If null, ssh_public_key_path is used."
  type        = string
  default     = null
}

variable "ssh_public_key_path" {
  description = "Path to an SSH public key used when ssh_public_key is null."
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "allowed_source_ip_ranges" {
  description = "Client public IP/CIDR ranges allowed to access SSH port 22 and Langfuse port 3000."
  type        = list(string)

  validation {
    condition     = length(var.allowed_source_ip_ranges) > 0
    error_message = "allowed_source_ip_ranges must include at least one trusted client IP/CIDR range."
  }
}

variable "vm_size" {
  description = "Azure VM size for the Langfuse demo VM."
  type        = string
  default     = "Standard_D4s_v5"
}

variable "os_disk_size_gb" {
  description = "OS disk size in GB."
  type        = number
  default     = 64
}

variable "vnet_address_space" {
  description = "Address space for the Langfuse VM virtual network."
  type        = list(string)
  default     = ["10.70.0.0/16"]
}

variable "subnet_address_prefixes" {
  description = "Address prefixes for the Langfuse VM subnet."
  type        = list(string)
  default     = ["10.70.1.0/24"]
}
