#!/usr/bin/env bash
set -euo pipefail

INFRA_DIR="environments/infra"
INDEX_NAME="${AZURE_AI_SEARCH_INDEX_NAME:-azure-agent-index}"
API_VERSION="${AZURE_AI_SEARCH_API_VERSION:-2023-11-01}"
SKIP_DOCUMENTS=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/setup-azure-ai-search.sh [options]

Options:
  --infra-dir <path>      Terraform core infra directory. Default: environments/infra
  --index-name <name>     Azure AI Search index name. Default: azure-agent-index
  --api-version <version> Azure AI Search API version. Default: 2023-11-01
  --skip-documents        Create/update the index only
  --dry-run               Print commands without creating the index or uploading documents
  -h, --help              Show this help

Environment overrides:
  AZURE_AI_SEARCH_ENDPOINT
  AZURE_AI_SEARCH_API_KEY
  AZURE_AI_SEARCH_INDEX_NAME
  AZURE_AI_SEARCH_API_VERSION
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --infra-dir)
      INFRA_DIR="$2"
      shift 2
      ;;
    --index-name)
      INDEX_NAME="$2"
      shift 2
      ;;
    --api-version)
      API_VERSION="$2"
      shift 2
      ;;
    --skip-documents)
      SKIP_DOCUMENTS=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
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

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[info] %s\n' "$*"
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

run() {
  if [ "$DRY_RUN" = true ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return
  fi

  "$@"
}

require_cmd terraform
require_cmd az
require_cmd jq
require_cmd uv

[ -d "$INFRA_DIR" ] || die "Terraform infra directory not found: $INFRA_DIR"

RESOURCE_GROUP_NAME="$(tf_raw resource_group_name)"
SEARCH_SERVICE_NAME="$(tf_jq resource_names '.ai_search')"
SEARCH_ENDPOINT="${AZURE_AI_SEARCH_ENDPOINT:-$(tf_raw search_endpoint)}"
SEARCH_API_KEY="${AZURE_AI_SEARCH_API_KEY:-}"

if [ -z "$SEARCH_API_KEY" ]; then
  if [ "$DRY_RUN" = true ]; then
    SEARCH_API_KEY="<azure-ai-search-api-key>"
  else
    log "Fetching Azure AI Search admin key from Azure"
    SEARCH_API_KEY="$(az search admin-key show \
      --resource-group "$RESOURCE_GROUP_NAME" \
      --service-name "$SEARCH_SERVICE_NAME" \
      --query primaryKey \
      -o tsv \
      --only-show-errors)"
  fi
fi

export AZURE_AI_SEARCH_ENDPOINT="$SEARCH_ENDPOINT"
export AZURE_AI_SEARCH_API_KEY="$SEARCH_API_KEY"
export AZURE_AI_SEARCH_INDEX_NAME="$INDEX_NAME"
export AZURE_AI_SEARCH_API_VERSION="$API_VERSION"

log "Creating/updating Azure AI Search index: $AZURE_AI_SEARCH_INDEX_NAME"
run uv run python examples/azure_ai_search/create_index.py

if [ "$SKIP_DOCUMENTS" = false ]; then
  log "Uploading sample Azure AI Search documents"
  run uv run python examples/azure_ai_search/create_document.py
fi

log "Azure AI Search setup complete."
