#!/usr/bin/env bash
set -euo pipefail

INFRA_DIR="environments/infra"
DRY_RUN=false
YES=false

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap-secrets.sh [options]

Options:
  --infra-dir <path>  Terraform core infra directory. Default: environments/infra
  --dry-run           Print the secrets that would be set without writing to Key Vault
  --yes               Use defaults and environment variables without interactive prompts
  -h, --help          Show this help

Environment overrides:
  Any Key Vault secret can be provided as an environment variable by replacing hyphens
  with underscores. Example:

    AZURE_OPENAI_API_VERSION=2024-12-01-preview
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...

  Additional supported inputs:

    POSTGRES_ADMIN_PASSWORD
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --infra-dir)
      INFRA_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --yes)
      YES=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

log() {
  printf '[info] %s\n' "$*"
}

warn() {
  printf '[warn] %s\n' "$*" >&2
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

tf_raw() {
  terraform -chdir="$INFRA_DIR" output -raw "$1"
}

tf_json() {
  terraform -chdir="$INFRA_DIR" output -json "$1"
}

tf_jq() {
  tf_json "$1" | jq -r "$2"
}

secret_to_env_name() {
  printf '%s' "$1" | tr '-' '_'
}

prompt_value() {
  local env_name="$1"
  local prompt="$2"
  local default="${3:-}"
  local sensitive="${4:-false}"
  local value="${!env_name:-}"

  if [ -n "$value" ]; then
    printf '%s' "$value"
    return
  fi

  if [ "$YES" = true ]; then
    [ -n "$default" ] || die "$env_name is required and has no default"
    printf '%s' "$default"
    return
  fi

  if [ "$sensitive" = true ]; then
    if [ -n "$default" ]; then
      printf '%s' "$prompt [default hidden]: " >&2
    else
      printf '%s' "$prompt: " >&2
    fi
    read -r -s value
    printf '\n' >&2
  else
    if [ -n "$default" ]; then
      printf '%s' "$prompt [$default]: " >&2
    else
      printf '%s' "$prompt: " >&2
    fi
    read -r value
  fi

  value="${value:-$default}"
  [ -n "$value" ] || die "$env_name is required"
  printf '%s' "$value"
}

secret_value() {
  local secret_name="$1"
  local prompt="$2"
  local default="${3:-}"
  local sensitive="${4:-false}"
  local env_name
  env_name="$(secret_to_env_name "$secret_name")"
  prompt_value "$env_name" "$prompt" "$default" "$sensitive"
}

set_secret() {
  local secret_name="$1"
  local value="$2"
  local source="$3"

  [ -n "$value" ] || die "Empty value for $secret_name"

  if [ "$DRY_RUN" = true ]; then
    printf '[dry-run] %s <- %s\n' "$secret_name" "$source"
    return
  fi

  az keyvault secret set \
    --vault-name "$KEY_VAULT_NAME" \
    --name "$secret_name" \
    --value "$value" \
    --only-show-errors \
    >/dev/null

  printf '[set] %s <- %s\n' "$secret_name" "$source"
}

try_az() {
  local value
  if value="$("$@" 2>/dev/null)"; then
    printf '%s' "$value"
  else
    printf ''
  fi
}

url_encode() {
  jq -sRr @uri
}

require_cmd terraform
require_cmd az
require_cmd jq

[ -d "$INFRA_DIR" ] || die "Terraform infra directory not found: $INFRA_DIR"

log "Reading Terraform outputs from $INFRA_DIR"

RESOURCE_GROUP_NAME="$(tf_raw resource_group_name)"
KEY_VAULT_NAME="$(tf_jq resource_names '.key_vault')"
STORAGE_ACCOUNT_NAME="$(tf_raw storage_account_name)"
SEARCH_SERVICE_NAME="$(tf_jq resource_names '.ai_search')"
OPENAI_ACCOUNT_NAME="$(tf_jq resource_names '.ai_services')"
CONTENT_SAFETY_ACCOUNT_NAME="$(tf_jq resource_names '.ai_content_safety')"

OPENAI_ENDPOINT="$(tf_raw openai_endpoint)"
SEARCH_ENDPOINT="$(tf_raw search_endpoint)"
CONTENT_SAFETY_ENDPOINT="$(tf_raw content_safety_endpoint)"

REDIS_HOST="$(tf_jq redis_memory '.host')"
REDIS_PORT="$(tf_jq redis_memory '.port | tostring')"
REDIS_DB="$(tf_jq redis_memory '.db')"
REDIS_STREAM_HOST="$(tf_jq redis_stream '.host')"
REDIS_STREAM_PORT="$(tf_jq redis_stream '.port | tostring')"

PYTHON_SESSION_ENDPOINT="$(tf_jq session_pool_endpoints '.python')"
BASH_SESSION_ENDPOINT="$(tf_jq session_pool_endpoints '.bash')"

POSTGRES_CONN_TEMPLATE="$(tf_jq postgres_connection_string_templates '.POSTGRES_CONN_STRING')"
POSTGRES_WEB_CONN_TEMPLATE="$(tf_jq postgres_connection_string_templates '.POSTGRES_WEB_CONN_STRING')"

OPENAI_MAIN_MODEL_DEFAULT="$(tf_jq openai_deployment_names '.main // empty')"
OPENAI_SMALL_MODEL_DEFAULT="$(tf_jq openai_deployment_names '.small // empty')"
OPENAI_EMBEDDING_MODEL_DEFAULT="$(tf_jq openai_deployment_names '.embedding // empty')"

OPENAI_MAIN_MODEL_DEFAULT="${OPENAI_MAIN_MODEL_DEFAULT:-gpt-5.4}"
OPENAI_SMALL_MODEL_DEFAULT="${OPENAI_SMALL_MODEL_DEFAULT:-gpt-5.4-nano}"
OPENAI_EMBEDDING_MODEL_DEFAULT="${OPENAI_EMBEDDING_MODEL_DEFAULT:-text-embedding-3-large}"

log "Using Key Vault: $KEY_VAULT_NAME"

log "Fetching Azure resource keys where possible"
BLOB_CONNECTION_STRING_AUTO="$(try_az az storage account show-connection-string --resource-group "$RESOURCE_GROUP_NAME" --name "$STORAGE_ACCOUNT_NAME" --query connectionString -o tsv --only-show-errors)"
OPENAI_API_KEY_AUTO="$(try_az az cognitiveservices account keys list --resource-group "$RESOURCE_GROUP_NAME" --name "$OPENAI_ACCOUNT_NAME" --query key1 -o tsv --only-show-errors)"
SEARCH_API_KEY_AUTO="$(try_az az search admin-key show --resource-group "$RESOURCE_GROUP_NAME" --service-name "$SEARCH_SERVICE_NAME" --query primaryKey -o tsv --only-show-errors)"
CONTENT_SAFETY_API_KEY_AUTO="$(try_az az cognitiveservices account keys list --resource-group "$RESOURCE_GROUP_NAME" --name "$CONTENT_SAFETY_ACCOUNT_NAME" --query key1 -o tsv --only-show-errors)"

[ -n "$BLOB_CONNECTION_STRING_AUTO" ] || warn "Could not auto-fetch Blob connection string; prompting instead"
[ -n "$OPENAI_API_KEY_AUTO" ] || warn "Could not auto-fetch Azure OpenAI API key; prompting instead"
[ -n "$SEARCH_API_KEY_AUTO" ] || warn "Could not auto-fetch Azure AI Search API key; prompting instead"
[ -n "$CONTENT_SAFETY_API_KEY_AUTO" ] || warn "Could not auto-fetch Azure AI Content Safety API key; prompting instead"

POSTGRES_ADMIN_PASSWORD_VALUE="$(prompt_value POSTGRES_ADMIN_PASSWORD "PostgreSQL administrator password" "" true)"
POSTGRES_ADMIN_PASSWORD_ENCODED="$(printf '%s' "$POSTGRES_ADMIN_PASSWORD_VALUE" | url_encode)"
POSTGRES_CONN_STRING_DEFAULT="${POSTGRES_CONN_TEMPLATE/<password>/$POSTGRES_ADMIN_PASSWORD_ENCODED}"
POSTGRES_WEB_CONN_STRING_DEFAULT="${POSTGRES_WEB_CONN_TEMPLATE/<password>/$POSTGRES_ADMIN_PASSWORD_ENCODED}"

set_secret "AZURE-OPENAI-ENDPOINT" "$OPENAI_ENDPOINT" "terraform output openai_endpoint"
set_secret "AZURE-OPENAI-API-KEY" "$(secret_value "AZURE-OPENAI-API-KEY" "Azure OpenAI API key" "$OPENAI_API_KEY_AUTO" true)" "az cognitiveservices account keys list or input"
set_secret "AZURE-OPENAI-API-VERSION" "$(secret_value "AZURE-OPENAI-API-VERSION" "Azure OpenAI API version" "2024-12-01-preview")" "input/default"
set_secret "AZURE-OPENAI-MAIN-MODEL" "$(secret_value "AZURE-OPENAI-MAIN-MODEL" "Azure OpenAI main deployment name" "$OPENAI_MAIN_MODEL_DEFAULT")" "terraform output/default"
set_secret "AZURE-OPENAI-SMALL-MODEL" "$(secret_value "AZURE-OPENAI-SMALL-MODEL" "Azure OpenAI small deployment name" "$OPENAI_SMALL_MODEL_DEFAULT")" "terraform output/default"
set_secret "AZURE-OPENAI-MAIN-MODEL-TIMEOUT" "$(secret_value "AZURE-OPENAI-MAIN-MODEL-TIMEOUT" "Azure OpenAI main model timeout seconds" "120")" "input/default"
set_secret "AZURE-OPENAI-SMALL-MODEL-TIMEOUT" "$(secret_value "AZURE-OPENAI-SMALL-MODEL-TIMEOUT" "Azure OpenAI small model timeout seconds" "60")" "input/default"
set_secret "AZURE-OPENAI-EMBEDDING-MODEL" "$(secret_value "AZURE-OPENAI-EMBEDDING-MODEL" "Azure OpenAI embedding deployment name" "$OPENAI_EMBEDDING_MODEL_DEFAULT")" "terraform output/default"
set_secret "AZURE-OPENAI-EMBEDDING-DIMS" "$(secret_value "AZURE-OPENAI-EMBEDDING-DIMS" "Azure OpenAI embedding dimensions" "3072")" "input/default"

set_secret "AZURE-AI-SEARCH-ENDPOINT" "$SEARCH_ENDPOINT" "terraform output search_endpoint"
set_secret "AZURE-AI-SEARCH-API-KEY" "$(secret_value "AZURE-AI-SEARCH-API-KEY" "Azure AI Search API key" "$SEARCH_API_KEY_AUTO" true)" "az search admin-key show or input"
set_secret "AZURE-AI-SEARCH-INDEX-NAME" "$(secret_value "AZURE-AI-SEARCH-INDEX-NAME" "Azure AI Search index name" "azure-agent-index")" "input/default"
set_secret "AZURE-AI-SEARCH-SEMANTIC-CONFIG" "$(secret_value "AZURE-AI-SEARCH-SEMANTIC-CONFIG" "Azure AI Search semantic config name" "default")" "input/default"
set_secret "AZURE-AI-SEARCH-API-VERSION" "$(secret_value "AZURE-AI-SEARCH-API-VERSION" "Azure AI Search API version" "2023-11-01")" "input/default"
set_secret "AZURE-AI-SEARCH-TOP-K" "$(secret_value "AZURE-AI-SEARCH-TOP-K" "Azure AI Search top K" "3")" "input/default"

set_secret "AZURE-AI-CONTENT-SAFETY-ENDPOINT" "$CONTENT_SAFETY_ENDPOINT" "terraform output content_safety_endpoint"
set_secret "AZURE-AI-CONTENT-SAFETY-API-KEY" "$(secret_value "AZURE-AI-CONTENT-SAFETY-API-KEY" "Azure AI Content Safety API key" "$CONTENT_SAFETY_API_KEY_AUTO" true)" "az cognitiveservices account keys list or input"

set_secret "AZURE-DYNAMIC-SESSIONS-PYTHON-POOL-ENDPOINT" "$PYTHON_SESSION_ENDPOINT" "terraform output session_pool_endpoints.python"
set_secret "AZURE-DYNAMIC-SESSIONS-BASH-POOL-ENDPOINT" "$BASH_SESSION_ENDPOINT" "terraform output session_pool_endpoints.bash"

set_secret "BLOB-CONNECTION-STRING" "$(secret_value "BLOB-CONNECTION-STRING" "Blob Storage connection string" "$BLOB_CONNECTION_STRING_AUTO" true)" "az storage account show-connection-string or input"

set_secret "REDIS-HOST" "$REDIS_HOST" "terraform output redis_memory.host"
set_secret "REDIS-USERNAME" "$(secret_value "REDIS-USERNAME" "Redis memory username" "default")" "input/default"
set_secret "REDIS-ACCESS-KEY" "$(secret_value "REDIS-ACCESS-KEY" "Redis memory access key" "" true)" "input"
set_secret "REDIS-PORT" "$REDIS_PORT" "terraform output redis_memory.port"
set_secret "REDIS-DB" "$REDIS_DB" "terraform output redis_memory.db"

set_secret "REDIS-STREAM-HOST" "$REDIS_STREAM_HOST" "terraform output redis_stream.host"
set_secret "REDIS-STREAM-USERNAME" "$(secret_value "REDIS-STREAM-USERNAME" "Redis stream username" "default")" "input/default"
set_secret "REDIS-STREAM-ACCESS-KEY" "$(secret_value "REDIS-STREAM-ACCESS-KEY" "Redis stream access key" "" true)" "input"
set_secret "REDIS-STREAM-PORT" "$REDIS_STREAM_PORT" "terraform output redis_stream.port"

set_secret "POSTGRES-CONN-STRING" "$(secret_value "POSTGRES-CONN-STRING" "PostgreSQL agent connection string" "$POSTGRES_CONN_STRING_DEFAULT" true)" "terraform output template + input password"
set_secret "POSTGRES-WEB-CONN-STRING" "$(secret_value "POSTGRES-WEB-CONN-STRING" "PostgreSQL web connection string" "$POSTGRES_WEB_CONN_STRING_DEFAULT" true)" "terraform output template + input password"

set_secret "LANGFUSE-PUBLIC-KEY" "$(secret_value "LANGFUSE-PUBLIC-KEY" "Langfuse public key")" "input"
set_secret "LANGFUSE-SECRET-KEY" "$(secret_value "LANGFUSE-SECRET-KEY" "Langfuse secret key" "" true)" "input"

log "Done. Key Vault secrets are ready for Container App secret references."
