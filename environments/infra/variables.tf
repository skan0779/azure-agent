variable "subscription_id" {
  description = "Azure subscription ID. Leave null to use ARM_SUBSCRIPTION_ID or the active Azure CLI subscription."
  type        = string
  default     = null
}

variable "project_name" {
  description = "Base project name used for Azure resource naming where the provider allows hyphens."
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

variable "location" {
  description = "Azure region for resources."
  type        = string
  default     = "koreacentral"
}

variable "name_suffix" {
  description = "Optional lowercase suffix for globally unique resource names. If empty, Terraform generates a stable random suffix."
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

variable "public_network_access_enabled" {
  description = "Public-first mode. Keep true for quick deployment without VNet/private endpoints."
  type        = bool
  default     = true
}

variable "allowed_ip_ranges" {
  description = "Optional public IP/CIDR allowlist for resources that support firewall rules. Empty means quickstart public mode."
  type        = list(string)
  default     = []
}

variable "log_analytics_retention_days" {
  description = "Log Analytics retention in days."
  type        = number
  default     = 30

  validation {
    condition     = var.log_analytics_retention_days >= 30 && var.log_analytics_retention_days <= 730
    error_message = "log_analytics_retention_days must be between 30 and 730."
  }
}

variable "container_registry_sku" {
  description = "Azure Container Registry SKU."
  type        = string
  default     = "Basic"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.container_registry_sku)
    error_message = "container_registry_sku must be Basic, Standard, or Premium."
  }
}

variable "key_vault_sku_name" {
  description = "Azure Key Vault SKU."
  type        = string
  default     = "standard"

  validation {
    condition     = contains(["standard", "premium"], var.key_vault_sku_name)
    error_message = "key_vault_sku_name must be standard or premium."
  }
}

variable "key_vault_soft_delete_retention_days" {
  description = "Key Vault soft delete retention in days."
  type        = number
  default     = 7

  validation {
    condition     = var.key_vault_soft_delete_retention_days >= 7 && var.key_vault_soft_delete_retention_days <= 90
    error_message = "key_vault_soft_delete_retention_days must be between 7 and 90."
  }
}

variable "key_vault_purge_protection_enabled" {
  description = "Enable Key Vault purge protection. Recommended for production; disabled by default for easy cleanup in dev."
  type        = bool
  default     = false
}

variable "assign_current_user_key_vault_secrets_officer" {
  description = "Assign Key Vault Secrets Officer to the current Terraform principal so application secrets can be set manually after deployment."
  type        = bool
  default     = true
}

variable "storage_account_replication_type" {
  description = "Storage Account replication type."
  type        = string
  default     = "LRS"

  validation {
    condition     = contains(["LRS", "GRS", "RAGRS", "ZRS", "GZRS", "RAGZRS"], var.storage_account_replication_type)
    error_message = "storage_account_replication_type must be one of LRS, GRS, RAGRS, ZRS, GZRS, or RAGZRS."
  }
}

variable "storage_blob_delete_retention_days" {
  description = "Blob delete retention in days."
  type        = number
  default     = 7

  validation {
    condition     = var.storage_blob_delete_retention_days >= 1 && var.storage_blob_delete_retention_days <= 365
    error_message = "storage_blob_delete_retention_days must be between 1 and 365."
  }
}

variable "storage_container_delete_retention_days" {
  description = "Blob container delete retention in days."
  type        = number
  default     = 7

  validation {
    condition     = var.storage_container_delete_retention_days >= 1 && var.storage_container_delete_retention_days <= 365
    error_message = "storage_container_delete_retention_days must be between 1 and 365."
  }
}

variable "search_sku" {
  description = "Azure AI Search SKU."
  type        = string
  default     = "basic"

  validation {
    condition     = contains(["free", "basic", "standard", "standard2", "standard3", "storage_optimized_l1", "storage_optimized_l2"], var.search_sku)
    error_message = "search_sku must be a supported Azure AI Search SKU."
  }
}

variable "search_replica_count" {
  description = "Azure AI Search replica count."
  type        = number
  default     = 1

  validation {
    condition     = var.search_replica_count >= 1
    error_message = "search_replica_count must be at least 1."
  }
}

variable "search_partition_count" {
  description = "Azure AI Search partition count."
  type        = number
  default     = 1

  validation {
    condition     = var.search_partition_count >= 1
    error_message = "search_partition_count must be at least 1."
  }
}

variable "openai_sku_name" {
  description = "Azure OpenAI account SKU."
  type        = string
  default     = "S0"
}

variable "create_openai_deployments" {
  description = "Create Azure OpenAI model deployments. Keep false when model availability/quota has not been confirmed for the selected region."
  type        = bool
  default     = false
}

variable "openai_deployments" {
  description = "Azure OpenAI deployments keyed by application role. Deployment names are what the app stores as AZURE-OPENAI-*-MODEL secrets."
  type = map(object({
    deployment_name = string
    model_name      = string
    model_version   = optional(string)
    sku_name        = optional(string, "GlobalStandard")
    sku_capacity    = optional(number, 1)
  }))
  default = {
    main = {
      deployment_name = "gpt-5.4"
      model_name      = "gpt-5.4"
    }
    small = {
      deployment_name = "gpt-5.4-nano"
      model_name      = "gpt-5.4-nano"
    }
    embedding = {
      deployment_name = "text-embedding-3-large"
      model_name      = "text-embedding-3-large"
    }
  }
}

variable "content_safety_sku_name" {
  description = "Azure AI Content Safety account SKU."
  type        = string
  default     = "S0"
}

variable "redis_memory_sku_name" {
  description = "Azure Managed Redis SKU for long-term/semantic memory features."
  type        = string
  default     = "Balanced_B0"
}

variable "redis_memory_high_availability_enabled" {
  description = "Enable high availability for the memory Redis cluster."
  type        = bool
  default     = false
}

variable "redis_memory_eviction_policy" {
  description = "Managed Redis memory database eviction policy."
  type        = string
  default     = "NoEviction"
}

variable "redis_memory_modules" {
  description = "Managed Redis modules required by the memory/checkpoint stack."
  type        = list(string)
  default     = ["RedisJSON", "RediSearch"]
}

variable "redis_stream_sku_name" {
  description = "Azure Managed Redis SKU used for streams, sessions, and SSE replay."
  type        = string
  default     = "Balanced_B0"
}

variable "redis_stream_high_availability_enabled" {
  description = "Enable high availability for the stream Redis cluster."
  type        = bool
  default     = false
}

variable "redis_stream_eviction_policy" {
  description = "Managed Redis stream database eviction policy."
  type        = string
  default     = "NoEviction"
}

variable "postgres_version" {
  description = "Azure Database for PostgreSQL Flexible Server version."
  type        = string
  default     = "16"
}

variable "postgres_administrator_login" {
  description = "PostgreSQL administrator login."
  type        = string
  default     = "azureagent"
}

variable "postgres_administrator_password" {
  description = "PostgreSQL administrator password. This value is stored in Terraform state because Terraform creates the server."
  type        = string
  sensitive   = true
}

variable "postgres_sku_name" {
  description = "PostgreSQL Flexible Server SKU."
  type        = string
  default     = "B_Standard_B1ms"
}

variable "postgres_storage_mb" {
  description = "PostgreSQL storage size in MB."
  type        = number
  default     = 32768
}

variable "postgres_backup_retention_days" {
  description = "PostgreSQL backup retention in days."
  type        = number
  default     = 7

  validation {
    condition     = var.postgres_backup_retention_days >= 7 && var.postgres_backup_retention_days <= 35
    error_message = "postgres_backup_retention_days must be between 7 and 35."
  }
}

variable "postgres_database_name" {
  description = "Database used by the Python agent/worker stack."
  type        = string
  default     = "azure_agent"
}

variable "postgres_web_database_name" {
  description = "Database used by the web gateway."
  type        = string
  default     = "azure_agent_web"
}

variable "postgres_allow_public_access_from_all_ips" {
  description = "Create a quickstart firewall rule allowing public access from all IPv4 addresses. Disable for restricted deployments."
  type        = bool
  default     = true
}

variable "postgres_firewall_rules" {
  description = "Additional PostgreSQL firewall rules keyed by rule name."
  type = map(object({
    start_ip_address = string
    end_ip_address   = string
  }))
  default = {}
}

variable "static_web_app_location" {
  description = "Azure Static Web Apps location. Static Web Apps supports a smaller set of regions than most Azure resources."
  type        = string
  default     = "eastasia"
}

variable "static_web_app_sku_tier" {
  description = "Azure Static Web Apps SKU tier."
  type        = string
  default     = "Free"
}

variable "static_web_app_sku_size" {
  description = "Azure Static Web Apps SKU size."
  type        = string
  default     = "Free"
}

variable "static_web_app_preview_environments_enabled" {
  description = "Enable preview environments for Azure Static Web Apps."
  type        = bool
  default     = true
}

variable "session_pool_cooldown_period_seconds" {
  description = "Container Apps Session Pool cooldown period in seconds."
  type        = number
  default     = 300
}

variable "session_pool_max_concurrent_sessions" {
  description = "Maximum concurrent sessions per Container Apps Session Pool."
  type        = number
  default     = 5
}

variable "session_pool_egress_enabled" {
  description = "Enable outbound network access from dynamic sessions."
  type        = bool
  default     = false
}

variable "deploy_container_apps" {
  description = "Create api, worker, and web Container Apps. Keep false until container images have been pushed to ACR."
  type        = bool
  default     = false
}

variable "deploy_container_app_job" {
  description = "Create the manual Container App Job. Keep false unless the job workflow is required."
  type        = bool
  default     = false
}

variable "api_min_replicas" {
  description = "Minimum replicas for the API Container App."
  type        = number
  default     = 1
}

variable "api_max_replicas" {
  description = "Maximum replicas for the API Container App."
  type        = number
  default     = 3
}

variable "api_cpu" {
  description = "CPU cores for the API container."
  type        = number
  default     = 0.5
}

variable "api_memory" {
  description = "Memory for the API container."
  type        = string
  default     = "1Gi"
}

variable "worker_min_replicas" {
  description = "Minimum replicas for the worker Container App."
  type        = number
  default     = 1
}

variable "worker_max_replicas" {
  description = "Maximum replicas for the worker Container App."
  type        = number
  default     = 3
}

variable "worker_cpu" {
  description = "CPU cores for the worker container."
  type        = number
  default     = 0.5
}

variable "worker_memory" {
  description = "Memory for the worker container."
  type        = string
  default     = "1Gi"
}

variable "web_min_replicas" {
  description = "Minimum replicas for the web Container App."
  type        = number
  default     = 1
}

variable "web_max_replicas" {
  description = "Maximum replicas for the web Container App."
  type        = number
  default     = 3
}

variable "web_cpu" {
  description = "CPU cores for the web container."
  type        = number
  default     = 0.5
}

variable "web_memory" {
  description = "Memory for the web container."
  type        = string
  default     = "1Gi"
}

variable "job_replica_timeout_in_seconds" {
  description = "Replica timeout for the manual Container App Job."
  type        = number
  default     = 1800
}

variable "job_replica_retry_limit" {
  description = "Replica retry limit for the manual Container App Job."
  type        = number
  default     = 1
}

variable "job_cpu" {
  description = "CPU cores for the manual job container."
  type        = number
  default     = 0.5
}

variable "job_memory" {
  description = "Memory for the manual job container."
  type        = string
  default     = "1Gi"
}

variable "sse_max_connection_seconds" {
  description = "API SSE maximum connection duration."
  type        = number
  default     = 600
}

variable "job_ttl_seconds" {
  description = "Job TTL in seconds."
  type        = number
  default     = 86400
}

variable "event_ttl_seconds" {
  description = "Event TTL in seconds."
  type        = number
  default     = 86400
}

variable "idempotency_ttl_seconds" {
  description = "Idempotency TTL in seconds."
  type        = number
  default     = 86400
}

variable "session_ttl_seconds" {
  description = "Reserved dynamic session TTL in seconds."
  type        = number
  default     = 3600
}

variable "session_reservation_ttl_seconds" {
  description = "Dynamic session reservation TTL in seconds."
  type        = number
  default     = 300
}

variable "session_lock_ttl_seconds" {
  description = "Dynamic session lock TTL in seconds."
  type        = number
  default     = 90
}

variable "worker_heartbeat_interval_seconds" {
  description = "Worker heartbeat interval in seconds."
  type        = number
  default     = 15
}

variable "worker_pending_claim_idle_ms" {
  description = "Worker pending claim idle threshold in milliseconds."
  type        = number
  default     = 300000
}

variable "worker_pending_claim_count" {
  description = "Number of pending Redis stream entries to claim per cycle."
  type        = number
  default     = 2
}

variable "worker_read_block_ms" {
  description = "Worker Redis stream read block duration in milliseconds."
  type        = number
  default     = 10000
}

variable "worker_read_count" {
  description = "Worker Redis stream read count."
  type        = number
  default     = 1
}

variable "worker_extra_env" {
  description = "Additional environment variables for worker and manual job containers."
  type        = map(string)
  default     = {}
}

variable "web_extra_env" {
  description = "Additional environment variables for the web container."
  type        = map(string)
  default     = {}
}

variable "api_worker_image_tag" {
  description = "Container image tag for azure-agent-api and azure-agent-worker. The ACR login server is added by the Container Apps resources."
  type        = string
  default     = "azure-agent:local"
}

variable "web_image_tag" {
  description = "Container image tag for azure-agent-web. The ACR login server is added by the Container Apps resources."
  type        = string
  default     = "azure-agent-web:local"
}
