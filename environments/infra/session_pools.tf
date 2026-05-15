resource "azapi_resource" "session_pool_python" {
  type      = "Microsoft.App/sessionPools@2025-10-02-preview"
  name      = local.names.session_pool_python
  parent_id = azurerm_resource_group.main.id
  location  = azurerm_resource_group.main.location
  tags      = local.tags

  body = {
    properties = {
      containerType = "PythonLTS"
      dynamicPoolConfiguration = {
        lifecycleConfiguration = {
          cooldownPeriodInSeconds = var.session_pool_cooldown_period_seconds
          lifecycleType           = "Timed"
        }
      }
      poolManagementType = "Dynamic"
      scaleConfiguration = {
        maxConcurrentSessions = var.session_pool_max_concurrent_sessions
      }
      sessionNetworkConfiguration = {
        status = var.session_pool_egress_enabled ? "EgressEnabled" : "EgressDisabled"
      }
    }
  }

  response_export_values = ["properties.poolManagementEndpoint"]
}

resource "azapi_resource" "session_pool_bash" {
  type      = "Microsoft.App/sessionPools@2025-10-02-preview"
  name      = local.names.session_pool_bash
  parent_id = azurerm_resource_group.main.id
  location  = azurerm_resource_group.main.location
  tags      = local.tags

  body = {
    properties = {
      containerType = "Shell"
      dynamicPoolConfiguration = {
        lifecycleConfiguration = {
          cooldownPeriodInSeconds = var.session_pool_cooldown_period_seconds
          lifecycleType           = "Timed"
        }
      }
      poolManagementType = "Dynamic"
      scaleConfiguration = {
        maxConcurrentSessions = var.session_pool_max_concurrent_sessions
      }
      sessionNetworkConfiguration = {
        status = var.session_pool_egress_enabled ? "EgressEnabled" : "EgressDisabled"
      }
    }
  }

  response_export_values = ["properties.poolManagementEndpoint"]
}
