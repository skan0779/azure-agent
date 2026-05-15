resource "azurerm_search_service" "main" {
  name                          = local.names.ai_search
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  sku                           = var.search_sku
  replica_count                 = var.search_replica_count
  partition_count               = var.search_partition_count
  local_authentication_enabled  = true
  public_network_access_enabled = var.public_network_access_enabled
  allowed_ips                   = var.allowed_ip_ranges
  tags                          = local.tags
}
