#!/usr/bin/env bash
set -euo pipefail

INFRA_DIR="environments/infra"
PLATFORM="linux/amd64"
API_WORKER_IMAGE_TAG="azure-agent:local"
WEB_IMAGE_TAG="azure-agent-web:local"
BUILD_API_WORKER=true
BUILD_WEB=true
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/build-push-images.sh [options]

Options:
  --infra-dir <path>          Terraform core infra directory. Default: environments/infra
  --platform <platform>       Docker build platform. Default: linux/amd64
  --api-worker-tag <tag>      API/worker/job image tag. Default: azure-agent:local
  --web-tag <tag>             Web image tag. Default: azure-agent-web:local
  --skip-api-worker           Do not build/push the API/worker/job image
  --skip-web                  Do not build/push the web image
  --dry-run                   Print commands without running Docker
  -h, --help                  Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --infra-dir)
      INFRA_DIR="$2"
      shift 2
      ;;
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    --api-worker-tag)
      API_WORKER_IMAGE_TAG="$2"
      shift 2
      ;;
    --web-tag)
      WEB_IMAGE_TAG="$2"
      shift 2
      ;;
    --skip-api-worker)
      BUILD_API_WORKER=false
      shift
      ;;
    --skip-web)
      BUILD_WEB=false
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

run() {
  if [ "$DRY_RUN" = true ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return
  fi

  "$@"
}

acr_name_from_login_server() {
  printf '%s' "$1" | cut -d'.' -f1
}

require_cmd terraform
require_cmd az
require_cmd docker

[ -d "$INFRA_DIR" ] || die "Terraform infra directory not found: $INFRA_DIR"

ACR_LOGIN_SERVER="$(tf_raw acr_login_server)"
ACR_NAME="$(acr_name_from_login_server "$ACR_LOGIN_SERVER")"

log "Using ACR: $ACR_LOGIN_SERVER"
run az acr login -n "$ACR_NAME"

if [ "$BUILD_API_WORKER" = true ]; then
  API_WORKER_IMAGE="$ACR_LOGIN_SERVER/$API_WORKER_IMAGE_TAG"
  run docker buildx build \
    --platform "$PLATFORM" \
    --provenance=false \
    -f environments/deploy/Dockerfile \
    -t "$API_WORKER_IMAGE" \
    --push .
fi

if [ "$BUILD_WEB" = true ]; then
  WEB_IMAGE="$ACR_LOGIN_SERVER/$WEB_IMAGE_TAG"
  run docker buildx build \
    --platform "$PLATFORM" \
    --provenance=false \
    -f apps/azure-agent-web/Dockerfile \
    -t "$WEB_IMAGE" \
    --push .
fi

log "Image build and push complete."
