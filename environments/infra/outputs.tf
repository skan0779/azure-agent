output "resource_group_name" {
  description = "Resource group created by this Terraform template."
  value       = azurerm_resource_group.main.name
}

output "location" {
  description = "Azure region used by this deployment."
  value       = azurerm_resource_group.main.location
}

output "log_analytics_workspace_id" {
  description = "Log Analytics Workspace resource ID."
  value       = azurerm_log_analytics_workspace.main.id
}

output "acr_login_server" {
  description = "Azure Container Registry login server."
  value       = azurerm_container_registry.main.login_server
}

output "key_vault_uri" {
  description = "Azure Key Vault URI."
  value       = azurerm_key_vault.main.vault_uri
}

output "storage_account_name" {
  description = "Storage Account name."
  value       = azurerm_storage_account.main.name
}

output "blob_container_names" {
  description = "Blob containers created for Azure Agent."
  value = {
    prompts = azurerm_storage_container.prompts.name
    files   = azurerm_storage_container.files.name
  }
}

output "search_endpoint" {
  description = "Azure AI Search endpoint."
  value       = azurerm_search_service.main.endpoint
}

output "openai_endpoint" {
  description = "Azure OpenAI endpoint."
  value       = azurerm_cognitive_account.openai.endpoint
}

output "openai_deployment_names" {
  description = "Azure OpenAI deployment names created when create_openai_deployments is true."
  value       = { for key, deployment in azurerm_cognitive_deployment.openai : key => deployment.name }
}

output "content_safety_endpoint" {
  description = "Azure AI Content Safety endpoint."
  value       = azurerm_cognitive_account.content_safety.endpoint
}

output "redis_memory" {
  description = "Azure Managed Redis connection metadata for memory/checkpoint features. Access keys are not output."
  value = {
    host = azurerm_managed_redis.memory.hostname
    port = azurerm_managed_redis.memory.default_database[0].port
    db   = "0"
  }
}

output "redis_stream" {
  description = "Azure Managed Redis connection metadata for streams/sessions. Access keys are not output."
  value = {
    host = azurerm_managed_redis.stream.hostname
    port = azurerm_managed_redis.stream.default_database[0].port
  }
}

output "postgres" {
  description = "PostgreSQL connection metadata. Password is not output."
  value = {
    fqdn           = azurerm_postgresql_flexible_server.main.fqdn
    admin_login    = var.postgres_administrator_login
    agent_database = azurerm_postgresql_flexible_server_database.agent.name
    web_database   = azurerm_postgresql_flexible_server_database.web.name
  }
}

output "postgres_connection_string_templates" {
  description = "Connection string templates for Key Vault secrets. Replace <password> before setting secrets."
  value = {
    POSTGRES_CONN_STRING     = "postgresql://${var.postgres_administrator_login}:<password>@${azurerm_postgresql_flexible_server.main.fqdn}:5432/${azurerm_postgresql_flexible_server_database.agent.name}?sslmode=require"
    POSTGRES_WEB_CONN_STRING = "postgresql://${var.postgres_administrator_login}:<password>@${azurerm_postgresql_flexible_server.main.fqdn}:5432/${azurerm_postgresql_flexible_server_database.web.name}?sslmode=require"
  }
}

output "container_apps_environment_id" {
  description = "Container Apps Environment resource ID."
  value       = azurerm_container_app_environment.main.id
}

output "session_pool_endpoints" {
  description = "Container Apps dynamic session pool management endpoints."
  value = {
    python = azapi_resource.session_pool_python.output.properties.poolManagementEndpoint
    bash   = azapi_resource.session_pool_bash.output.properties.poolManagementEndpoint
  }
}

output "container_app_urls" {
  description = "Container App public URLs. Null until deploy_container_apps is true."
  value = {
    api = var.deploy_container_apps ? "https://${azurerm_container_app.api[0].ingress[0].fqdn}" : null
    web = var.deploy_container_apps ? "https://${azurerm_container_app.web[0].ingress[0].fqdn}" : null
  }
}

output "static_web_app_url" {
  description = "Azure Static Web Apps public URL."
  value       = "https://${azurerm_static_web_app.ui.default_host_name}"
}

output "static_web_app_api_key" {
  description = "Azure Static Web Apps deployment token. Store it as AZURE_STATIC_WEB_APPS_API_TOKEN in GitHub Actions."
  value       = azurerm_static_web_app.ui.api_key
  sensitive   = true
}

output "github_actions_variables" {
  description = "Values typically used by the UI deployment workflow."
  value = {
    NEXT_PUBLIC_AGENT_WEB_URL = var.deploy_container_apps ? "https://${azurerm_container_app.web[0].ingress[0].fqdn}" : "<set-after-web-container-app-deployment>"
  }
}

output "key_vault_secret_names" {
  description = "Secret names expected by the application. Values are set manually after deployment."
  value       = local.key_vault_secret_names
}

output "resource_names" {
  description = "Planned resource names for the public-first Azure Agent deployment."
  value       = local.names
}

output "network_access_mode" {
  description = "Current network access profile derived from allowed_ip_ranges."
  value       = local.network_access_mode
}
