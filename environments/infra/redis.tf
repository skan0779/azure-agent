resource "azurerm_managed_redis" "memory" {
  name                      = local.names.redis_memory
  resource_group_name       = azurerm_resource_group.main.name
  location                  = azurerm_resource_group.main.location
  sku_name                  = var.redis_memory_sku_name
  high_availability_enabled = var.redis_memory_high_availability_enabled
  public_network_access     = var.public_network_access_enabled ? "Enabled" : "Disabled"
  tags                      = local.tags

  default_database {
    access_keys_authentication_enabled = true
    client_protocol                    = "Encrypted"
    clustering_policy                  = "EnterpriseCluster"
    eviction_policy                    = var.redis_memory_eviction_policy

    dynamic "module" {
      for_each = var.redis_memory_modules

      content {
        name = module.value
      }
    }
  }
}

resource "azurerm_managed_redis" "stream" {
  name                      = local.names.redis_stream
  resource_group_name       = azurerm_resource_group.main.name
  location                  = azurerm_resource_group.main.location
  sku_name                  = var.redis_stream_sku_name
  high_availability_enabled = var.redis_stream_high_availability_enabled
  public_network_access     = var.public_network_access_enabled ? "Enabled" : "Disabled"
  tags                      = local.tags

  default_database {
    access_keys_authentication_enabled = true
    client_protocol                    = "Encrypted"
    clustering_policy                  = "OSSCluster"
    eviction_policy                    = var.redis_stream_eviction_policy
  }
}
