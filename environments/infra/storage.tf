resource "azurerm_storage_account" "main" {
  name                            = local.names.storage_account
  resource_group_name             = azurerm_resource_group.main.name
  location                        = azurerm_resource_group.main.location
  account_tier                    = "Standard"
  account_replication_type        = var.storage_account_replication_type
  account_kind                    = "StorageV2"
  access_tier                     = "Hot"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = true
  public_network_access_enabled   = var.public_network_access_enabled
  tags                            = local.tags

  blob_properties {
    delete_retention_policy {
      days = var.storage_blob_delete_retention_days
    }

    container_delete_retention_policy {
      days = var.storage_container_delete_retention_days
    }
  }
}

resource "azurerm_storage_account_network_rules" "main" {
  storage_account_id = azurerm_storage_account.main.id
  default_action     = local.network_default_action
  bypass             = ["AzureServices"]
  ip_rules           = var.allowed_ip_ranges
}

resource "azurerm_storage_container" "prompts" {
  name                  = local.names.prompt_blob_container
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"

  depends_on = [
    azurerm_storage_account_network_rules.main,
  ]
}

resource "azurerm_storage_container" "files" {
  name                  = local.names.files_blob_container
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"

  depends_on = [
    azurerm_storage_account_network_rules.main,
  ]
}
