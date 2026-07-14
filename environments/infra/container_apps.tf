resource "azurerm_container_app_environment" "main" {
  name                       = local.names.container_apps_environment
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  logs_destination           = "log-analytics"
  public_network_access      = var.public_network_access_enabled ? "Enabled" : "Disabled"
  tags                       = local.tags
}

resource "azurerm_user_assigned_identity" "api" {
  name                = local.names.identity_api
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

resource "azurerm_user_assigned_identity" "worker" {
  name                = local.names.identity_worker
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

resource "azurerm_user_assigned_identity" "web" {
  name                = local.names.identity_web
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

resource "azurerm_user_assigned_identity" "job" {
  name                = local.names.identity_job
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = local.tags
}

resource "azurerm_container_app" "api" {
  count                        = var.deploy_container_apps ? 1 : 0
  name                         = local.names.container_app_api
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.api.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.api.id
  }

  dynamic "secret" {
    for_each = local.api_secret_env

    content {
      name                = secret.value.secret_name
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/${secret.value.key_vault_secret_name}"
      identity            = azurerm_user_assigned_identity.api.id
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8080
    transport        = "http"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = var.api_min_replicas
    max_replicas = var.api_max_replicas

    container {
      name   = "api"
      image  = "${azurerm_container_registry.main.login_server}/${var.api_worker_image_tag}"
      cpu    = var.api_cpu
      memory = var.api_memory

      dynamic "env" {
        for_each = local.api_env

        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.api_secret_env

        content {
          name        = env.key
          secret_name = env.value.secret_name
        }
      }
    }
  }

  depends_on = [
    azurerm_role_assignment.api_acr_pull,
    azurerm_role_assignment.api_key_vault_secrets_user,
  ]
}

resource "azurerm_container_app" "worker" {
  count                        = var.deploy_container_apps ? 1 : 0
  name                         = local.names.container_app_worker
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.worker.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.worker.id
  }

  dynamic "secret" {
    for_each = local.worker_secret_env

    content {
      name                = secret.value.secret_name
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/${secret.value.key_vault_secret_name}"
      identity            = azurerm_user_assigned_identity.worker.id
    }
  }

  template {
    min_replicas = var.worker_min_replicas
    max_replicas = var.worker_max_replicas

    container {
      name    = "worker"
      image   = "${azurerm_container_registry.main.login_server}/${var.api_worker_image_tag}"
      command = ["sh"]
      args    = ["-lc", "uv run azure-agent-worker"]
      cpu     = var.worker_cpu
      memory  = var.worker_memory

      dynamic "env" {
        for_each = local.worker_env

        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.worker_secret_env

        content {
          name        = env.key
          secret_name = env.value.secret_name
        }
      }
    }
  }

  depends_on = [
    azurerm_role_assignment.worker_acr_pull,
    azurerm_role_assignment.worker_key_vault_secrets_user,
    azurerm_role_assignment.worker_storage_blob_data_reader,
    azurerm_role_assignment.worker_python_session_executor,
    azurerm_role_assignment.worker_bash_session_executor,
  ]
}

resource "azurerm_container_app" "web" {
  count                        = var.deploy_container_apps ? 1 : 0
  name                         = local.names.container_app_web
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.web.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.web.id
  }

  dynamic "secret" {
    for_each = local.web_secret_env

    content {
      name                = secret.value.secret_name
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/${secret.value.key_vault_secret_name}"
      identity            = azurerm_user_assigned_identity.web.id
    }
  }

  ingress {
    external_enabled = true
    target_port      = 3001
    transport        = "http"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = var.web_min_replicas
    max_replicas = var.web_max_replicas

    container {
      name   = "web"
      image  = "${azurerm_container_registry.main.login_server}/${var.web_image_tag}"
      cpu    = var.web_cpu
      memory = var.web_memory

      dynamic "env" {
        for_each = local.web_env

        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.web_secret_env

        content {
          name        = env.key
          secret_name = env.value.secret_name
        }
      }
    }
  }

  depends_on = [
    azurerm_container_app.api,
    azurerm_role_assignment.web_acr_pull,
    azurerm_role_assignment.web_key_vault_secrets_user,
  ]
}

resource "azurerm_container_app_job" "manual" {
  count                        = var.deploy_container_app_job ? 1 : 0
  name                         = local.names.container_app_job
  resource_group_name          = azurerm_resource_group.main.name
  location                     = azurerm_resource_group.main.location
  container_app_environment_id = azurerm_container_app_environment.main.id
  replica_timeout_in_seconds   = var.job_replica_timeout_in_seconds
  replica_retry_limit          = var.job_replica_retry_limit
  tags                         = local.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.job.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.job.id
  }

  dynamic "secret" {
    for_each = local.job_secret_env

    content {
      name                = secret.value.secret_name
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/${secret.value.key_vault_secret_name}"
      identity            = azurerm_user_assigned_identity.job.id
    }
  }

  manual_trigger_config {
    parallelism              = 1
    replica_completion_count = 1
  }

  template {
    container {
      name    = "job"
      image   = "${azurerm_container_registry.main.login_server}/${var.api_worker_image_tag}"
      command = ["sh"]
      args    = ["-lc", "uv run --no-sync alembic upgrade head"]
      cpu     = var.job_cpu
      memory  = var.job_memory

      dynamic "env" {
        for_each = local.job_secret_env

        content {
          name        = env.key
          secret_name = env.value.secret_name
        }
      }
    }
  }

  depends_on = [
    azurerm_role_assignment.job_acr_pull,
    azurerm_role_assignment.job_key_vault_secrets_user,
    azurerm_role_assignment.job_storage_blob_data_reader,
    azurerm_role_assignment.job_python_session_executor,
    azurerm_role_assignment.job_bash_session_executor,
  ]
}
