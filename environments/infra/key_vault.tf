resource "azurerm_key_vault" "main" {
  name                          = local.names.key_vault
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  tenant_id                     = data.azurerm_client_config.current.tenant_id
  sku_name                      = var.key_vault_sku_name
  rbac_authorization_enabled    = true
  public_network_access_enabled = var.public_network_access_enabled
  soft_delete_retention_days    = var.key_vault_soft_delete_retention_days
  purge_protection_enabled      = var.key_vault_purge_protection_enabled
  tags                          = local.tags

  network_acls {
    bypass         = "AzureServices"
    default_action = local.network_default_action
    ip_rules       = var.allowed_ip_ranges
  }
}

resource "azurerm_role_assignment" "current_key_vault_secrets_officer" {
  count                = var.assign_current_user_key_vault_secrets_officer ? 1 : 0
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}
