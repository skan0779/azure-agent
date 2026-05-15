resource "azurerm_container_registry" "main" {
  name                          = local.names.container_registry
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  sku                           = var.container_registry_sku
  admin_enabled                 = false
  public_network_access_enabled = var.public_network_access_enabled
  tags                          = local.tags
}
