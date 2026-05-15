resource "azurerm_cognitive_account" "openai" {
  name                          = local.names.ai_services
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  kind                          = "OpenAI"
  sku_name                      = var.openai_sku_name
  custom_subdomain_name         = local.names.ai_services
  local_auth_enabled            = true
  public_network_access_enabled = var.public_network_access_enabled
  tags                          = local.tags

  network_acls {
    default_action = local.network_default_action
    ip_rules       = var.allowed_ip_ranges
  }
}

resource "azurerm_cognitive_deployment" "openai" {
  for_each = var.create_openai_deployments ? var.openai_deployments : {}

  name                 = each.value.deployment_name
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = each.value.model_name
    version = each.value.model_version
  }

  sku {
    name     = each.value.sku_name
    capacity = each.value.sku_capacity
  }
}

resource "azurerm_cognitive_account" "content_safety" {
  name                          = local.names.ai_content_safety
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  kind                          = "ContentSafety"
  sku_name                      = var.content_safety_sku_name
  custom_subdomain_name         = local.names.ai_content_safety
  local_auth_enabled            = true
  public_network_access_enabled = var.public_network_access_enabled
  tags                          = local.tags

  network_acls {
    default_action = local.network_default_action
    ip_rules       = var.allowed_ip_ranges
  }
}
