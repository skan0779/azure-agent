resource "azurerm_static_web_app" "ui" {
  name                          = local.names.static_web_app
  resource_group_name           = azurerm_resource_group.main.name
  location                      = var.static_web_app_location
  sku_tier                      = var.static_web_app_sku_tier
  sku_size                      = var.static_web_app_sku_size
  public_network_access_enabled = var.public_network_access_enabled
  preview_environments_enabled  = var.static_web_app_preview_environments_enabled
  tags                          = local.tags
}
