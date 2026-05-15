resource "azurerm_role_assignment" "api_acr_pull" {
  scope                            = azurerm_container_registry.main.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_user_assigned_identity.api.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "worker_acr_pull" {
  scope                            = azurerm_container_registry.main.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_user_assigned_identity.worker.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "web_acr_pull" {
  scope                            = azurerm_container_registry.main.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_user_assigned_identity.web.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "job_acr_pull" {
  scope                            = azurerm_container_registry.main.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_user_assigned_identity.job.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "api_key_vault_secrets_user" {
  scope                            = azurerm_key_vault.main.id
  role_definition_name             = "Key Vault Secrets User"
  principal_id                     = azurerm_user_assigned_identity.api.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "worker_key_vault_secrets_user" {
  scope                            = azurerm_key_vault.main.id
  role_definition_name             = "Key Vault Secrets User"
  principal_id                     = azurerm_user_assigned_identity.worker.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "web_key_vault_secrets_user" {
  scope                            = azurerm_key_vault.main.id
  role_definition_name             = "Key Vault Secrets User"
  principal_id                     = azurerm_user_assigned_identity.web.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "job_key_vault_secrets_user" {
  scope                            = azurerm_key_vault.main.id
  role_definition_name             = "Key Vault Secrets User"
  principal_id                     = azurerm_user_assigned_identity.job.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "worker_storage_blob_data_reader" {
  scope                            = azurerm_storage_account.main.id
  role_definition_name             = "Storage Blob Data Reader"
  principal_id                     = azurerm_user_assigned_identity.worker.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "job_storage_blob_data_reader" {
  scope                            = azurerm_storage_account.main.id
  role_definition_name             = "Storage Blob Data Reader"
  principal_id                     = azurerm_user_assigned_identity.job.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "worker_python_session_executor" {
  scope                            = azapi_resource.session_pool_python.id
  role_definition_name             = "Azure ContainerApps Session Executor"
  principal_id                     = azurerm_user_assigned_identity.worker.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "worker_bash_session_executor" {
  scope                            = azapi_resource.session_pool_bash.id
  role_definition_name             = "Azure ContainerApps Session Executor"
  principal_id                     = azurerm_user_assigned_identity.worker.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "job_python_session_executor" {
  scope                            = azapi_resource.session_pool_python.id
  role_definition_name             = "Azure ContainerApps Session Executor"
  principal_id                     = azurerm_user_assigned_identity.job.principal_id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "job_bash_session_executor" {
  scope                            = azapi_resource.session_pool_bash.id
  role_definition_name             = "Azure ContainerApps Session Executor"
  principal_id                     = azurerm_user_assigned_identity.job.principal_id
  skip_service_principal_aad_check = true
}
