locals {
  base_name      = "${var.project_name}-${var.environment}"
  compact_base   = replace(local.base_name, "-", "")
  suffix         = var.name_suffix != "" ? var.name_suffix : random_string.suffix.result
  dashed_suffix  = "-${local.suffix}"
  compact_suffix = local.suffix

  names = {
    resource_group             = "${local.base_name}${local.dashed_suffix}-rg"
    log_analytics_workspace    = "${local.base_name}${local.dashed_suffix}-law"
    container_registry         = substr("${local.compact_base}${local.compact_suffix}acr", 0, 50)
    key_vault                  = substr("${local.base_name}${local.dashed_suffix}-kv", 0, 24)
    storage_account            = substr("${local.compact_base}${local.compact_suffix}st", 0, 24)
    ai_search                  = substr("${local.base_name}${local.dashed_suffix}-search", 0, 60)
    ai_services                = substr("${local.base_name}${local.dashed_suffix}-ai", 0, 64)
    ai_content_safety          = substr("${local.base_name}${local.dashed_suffix}-safety", 0, 64)
    redis_memory               = substr("${local.base_name}${local.dashed_suffix}-redis-memory", 0, 63)
    redis_stream               = substr("${local.base_name}${local.dashed_suffix}-redis-stream", 0, 63)
    postgres                   = substr("${local.base_name}${local.dashed_suffix}-pg", 0, 63)
    container_apps_environment = "${local.base_name}${local.dashed_suffix}-cae"
    container_app_api          = "${local.base_name}${local.dashed_suffix}-api"
    container_app_worker       = "${local.base_name}${local.dashed_suffix}-worker"
    container_app_web          = "${local.base_name}${local.dashed_suffix}-web"
    container_app_job          = "${local.base_name}${local.dashed_suffix}-job"
    identity_api               = "${local.base_name}${local.dashed_suffix}-api-id"
    identity_worker            = "${local.base_name}${local.dashed_suffix}-worker-id"
    identity_web               = "${local.base_name}${local.dashed_suffix}-web-id"
    identity_job               = "${local.base_name}${local.dashed_suffix}-job-id"
    session_pool_python        = substr(replace("${local.base_name}${local.dashed_suffix}python", "-", ""), 0, 63)
    session_pool_bash          = substr(replace("${local.base_name}${local.dashed_suffix}bash", "-", ""), 0, 63)
    static_web_app             = substr("${local.base_name}${local.dashed_suffix}-ui", 0, 60)
    prompt_blob_container      = "prompts"
    files_blob_container       = "files"
  }

  network_access_mode = length(var.allowed_ip_ranges) > 0 ? "restricted-public" : "public"
  network_default_action = (
    var.public_network_access_enabled && length(var.allowed_ip_ranges) > 0
    ? "Deny"
    : "Allow"
  )

  postgres_firewall_rules = merge(
    var.postgres_allow_public_access_from_all_ips ? {
      allow-all = {
        start_ip_address = "0.0.0.0"
        end_ip_address   = "255.255.255.255"
      }
    } : {},
    var.postgres_firewall_rules,
  )

  api_env = {
    SSE_MAX_CONNECTION_SECONDS      = tostring(var.sse_max_connection_seconds)
    JOB_TTL_SECONDS                 = tostring(var.job_ttl_seconds)
    EVENT_TTL_SECONDS               = tostring(var.event_ttl_seconds)
    IDEMPOTENCY_TTL_SECONDS         = tostring(var.idempotency_ttl_seconds)
    SESSION_TTL_SECONDS             = tostring(var.session_ttl_seconds)
    SESSION_RESERVATION_TTL_SECONDS = tostring(var.session_reservation_ttl_seconds)
    SESSION_LOCK_TTL_SECONDS        = tostring(var.session_lock_ttl_seconds)
  }

  worker_env = merge(
    {
      JOB_TTL_SECONDS                   = tostring(var.job_ttl_seconds)
      EVENT_TTL_SECONDS                 = tostring(var.event_ttl_seconds)
      IDEMPOTENCY_TTL_SECONDS           = tostring(var.idempotency_ttl_seconds)
      SESSION_TTL_SECONDS               = tostring(var.session_ttl_seconds)
      SESSION_RESERVATION_TTL_SECONDS   = tostring(var.session_reservation_ttl_seconds)
      SESSION_LOCK_TTL_SECONDS          = tostring(var.session_lock_ttl_seconds)
      WORKER_HEARTBEAT_INTERVAL_SECONDS = tostring(var.worker_heartbeat_interval_seconds)
      WORKER_PENDING_CLAIM_IDLE_MS      = tostring(var.worker_pending_claim_idle_ms)
      WORKER_PENDING_CLAIM_COUNT        = tostring(var.worker_pending_claim_count)
      WORKER_READ_BLOCK_MS              = tostring(var.worker_read_block_ms)
      WORKER_READ_COUNT                 = tostring(var.worker_read_count)
    },
    var.worker_extra_env,
  )

  web_env = merge(
    {
      AGENT_API_BASE_URL = var.deploy_container_apps ? "https://${azurerm_container_app.api[0].ingress[0].fqdn}" : ""
      CORS_ORIGINS       = "https://${azurerm_static_web_app.ui.default_host_name}"
      HOST               = "0.0.0.0"
      PORT               = "3001"
    },
    var.web_extra_env,
  )

  api_secret_env_names = [
    "BLOB_CONNECTION_STRING",
    "POSTGRES_WEB_CONN_STRING",
    "REDIS_STREAM_HOST",
    "REDIS_STREAM_USERNAME",
    "REDIS_STREAM_ACCESS_KEY",
    "REDIS_STREAM_PORT",
  ]

  worker_secret_env_names = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_MAIN_MODEL",
    "AZURE_OPENAI_MAIN_MODEL_TIMEOUT",
    "AZURE_OPENAI_SMALL_MODEL",
    "AZURE_OPENAI_SMALL_MODEL_TIMEOUT",
    "AZURE_OPENAI_EMBEDDING_MODEL",
    "AZURE_OPENAI_EMBEDDING_DIMS",
    "AZURE_AI_SEARCH_ENDPOINT",
    "AZURE_AI_SEARCH_API_KEY",
    "AZURE_AI_SEARCH_INDEX_NAME",
    "AZURE_AI_SEARCH_SEMANTIC_CONFIG",
    "AZURE_AI_SEARCH_API_VERSION",
    "AZURE_AI_SEARCH_TOP_K",
    "AZURE_AI_CONTENT_SAFETY_ENDPOINT",
    "AZURE_AI_CONTENT_SAFETY_API_KEY",
    "AZURE_DYNAMIC_SESSIONS_PYTHON_POOL_ENDPOINT",
    "AZURE_DYNAMIC_SESSIONS_BASH_POOL_ENDPOINT",
    "BLOB_CONNECTION_STRING",
    "REDIS_HOST",
    "REDIS_USERNAME",
    "REDIS_ACCESS_KEY",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_STREAM_HOST",
    "REDIS_STREAM_USERNAME",
    "REDIS_STREAM_ACCESS_KEY",
    "REDIS_STREAM_PORT",
    "POSTGRES_CONN_STRING",
    "POSTGRES_WEB_CONN_STRING",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
  ]

  web_secret_env_names = [
    "POSTGRES_WEB_CONN_STRING",
  ]

  job_secret_env_names = [
    "POSTGRES_WEB_CONN_STRING",
  ]

  api_secret_env = {
    for name in local.api_secret_env_names : name => {
      secret_name           = lower(replace(name, "_", "-"))
      key_vault_secret_name = replace(name, "_", "-")
    }
  }

  worker_secret_env = {
    for name in local.worker_secret_env_names : name => {
      secret_name           = lower(replace(name, "_", "-"))
      key_vault_secret_name = replace(name, "_", "-")
    }
  }

  web_secret_env = {
    for name in local.web_secret_env_names : name => {
      secret_name           = lower(replace(name, "_", "-"))
      key_vault_secret_name = replace(name, "_", "-")
    }
  }

  job_secret_env = {
    for name in local.job_secret_env_names : name => {
      secret_name           = lower(replace(name, "_", "-"))
      key_vault_secret_name = replace(name, "_", "-")
    }
  }

  key_vault_secret_names = [
    "AZURE-OPENAI-ENDPOINT",
    "AZURE-OPENAI-API-KEY",
    "AZURE-OPENAI-API-VERSION",
    "AZURE-OPENAI-MAIN-MODEL",
    "AZURE-OPENAI-SMALL-MODEL",
    "AZURE-OPENAI-MAIN-MODEL-TIMEOUT",
    "AZURE-OPENAI-SMALL-MODEL-TIMEOUT",
    "AZURE-OPENAI-EMBEDDING-MODEL",
    "AZURE-OPENAI-EMBEDDING-DIMS",
    "AZURE-AI-SEARCH-ENDPOINT",
    "AZURE-AI-SEARCH-API-KEY",
    "AZURE-AI-SEARCH-INDEX-NAME",
    "AZURE-AI-SEARCH-SEMANTIC-CONFIG",
    "AZURE-AI-SEARCH-API-VERSION",
    "AZURE-AI-SEARCH-TOP-K",
    "AZURE-AI-CONTENT-SAFETY-ENDPOINT",
    "AZURE-AI-CONTENT-SAFETY-API-KEY",
    "AZURE-DYNAMIC-SESSIONS-PYTHON-POOL-ENDPOINT",
    "AZURE-DYNAMIC-SESSIONS-BASH-POOL-ENDPOINT",
    "BLOB-CONNECTION-STRING",
    "REDIS-HOST",
    "REDIS-USERNAME",
    "REDIS-ACCESS-KEY",
    "REDIS-PORT",
    "REDIS-DB",
    "REDIS-STREAM-HOST",
    "REDIS-STREAM-USERNAME",
    "REDIS-STREAM-ACCESS-KEY",
    "REDIS-STREAM-PORT",
    "POSTGRES-CONN-STRING",
    "POSTGRES-WEB-CONN-STRING",
    "LANGFUSE-SECRET-KEY",
    "LANGFUSE-PUBLIC-KEY",
  ]

  tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags,
  )
}
