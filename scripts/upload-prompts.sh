#!/usr/bin/env bash
set -euo pipefail

INFRA_DIR="environments/infra"
PROMPT_DIR="src/azure_agent/prompts"
DRY_RUN=false
CONNECTION_STRING="${BLOB_CONNECTION_STRING:-}"

usage() {
  cat <<'EOF'
Usage: scripts/upload-prompts.sh [options]

Options:
  --infra-dir <path>          Terraform core infra directory. Default: environments/infra
  --prompt-dir <path>         Prompt directory. Default: src/azure_agent/prompts
  --connection-string <value> Storage connection string override
  --dry-run                   Print uploads without writing to Blob Storage
  -h, --help                  Show this help

Environment overrides:
  BLOB_CONNECTION_STRING      Storage connection string override
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --infra-dir)
      INFRA_DIR="$2"
      shift 2
      ;;
    --prompt-dir)
      PROMPT_DIR="$2"
      shift 2
      ;;
    --connection-string)
      CONNECTION_STRING="$2"
      shift 2
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

upload_prompt() {
  local file_name="$1"
  local file_path="$PROMPT_DIR/$file_name"

  [ -f "$file_path" ] || die "Prompt file not found: $file_path"

  if [ "$DRY_RUN" = true ]; then
    printf '[dry-run] upload %s -> %s/%s\n' "$file_path" "$PROMPTS_CONTAINER_NAME" "$file_name"
    return
  fi

  az storage blob upload \
    --connection-string "$CONNECTION_STRING" \
    --container-name "$PROMPTS_CONTAINER_NAME" \
    --file "$file_path" \
    --name "$file_name" \
    --overwrite \
    --only-show-errors \
    >/dev/null

  printf '[upload] %s -> %s/%s\n' "$file_path" "$PROMPTS_CONTAINER_NAME" "$file_name"
}

require_cmd terraform
require_cmd az
require_cmd jq

[ -d "$INFRA_DIR" ] || die "Terraform infra directory not found: $INFRA_DIR"
[ -d "$PROMPT_DIR" ] || die "Prompt directory not found: $PROMPT_DIR"

RESOURCE_GROUP_NAME="$(tf_raw resource_group_name)"
STORAGE_ACCOUNT_NAME="$(tf_raw storage_account_name)"
PROMPTS_CONTAINER_NAME="$(tf_jq blob_container_names '.prompts')"

if [ -z "$CONNECTION_STRING" ]; then
  log "Fetching Storage connection string from Azure"
  CONNECTION_STRING="$(az storage account show-connection-string \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --name "$STORAGE_ACCOUNT_NAME" \
    --query connectionString \
    -o tsv \
    --only-show-errors)"
fi

upload_prompt "main_agent.yaml"
upload_prompt "sandbox_agent.yaml"

log "Prompt upload complete."
