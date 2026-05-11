terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "resource_group_name" {
  type = string
}

variable "location" {
  type = string
}

variable "container_app_environment_id" {
  type = string
}

variable "migration_job_name" {
  type = string
}

variable "image" {
  type = string
  # example: myacr.azurecr.io/azure-agent:20260507
}

variable "acr_login_server" {
  type = string
  # example: myacr.azurecr.io
}

variable "key_vault_url" {
  type = string
}

variable "tags" {
  type = map(string)
  default = {
    Owner = "seokhwan.jung@kt.com"
  }
}

resource "azurerm_container_app_job" "migration" {
  name                         = var.migration_job_name
  location                     = var.location
  resource_group_name          = var.resource_group_name
  container_app_environment_id = var.container_app_environment_id

  workload_profile_name        = "Consumption"
  replica_timeout_in_seconds   = 1800
  replica_retry_limit          = 0

  tags = var.tags

  identity {
    type = "SystemAssigned"
  }

  registry {
    server   = var.acr_login_server
    identity = "System"
  }

  manual_trigger_config {
    replica_completion_count = 1
    parallelism              = 1
  }

  template {
    container {
      name   = "migration"
      image  = var.image
      cpu    = 0.5
      memory = "1Gi"

      command = ["uv"]
      args    = ["run", "alembic", "upgrade", "head"]

      env {
        name  = "KEY_VAULT_URL"
        value = var.key_vault_url
      }
    }
  }
}