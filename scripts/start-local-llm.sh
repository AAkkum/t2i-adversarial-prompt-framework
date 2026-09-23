#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8082}"
LLM_ALIAS="${LLM_ALIAS:-local-llm}"
LLM_MODEL="${LLM_MODEL:-}"
LLM_MMPROJ="${LLM_MMPROJ:-}"
LLM_CONTEXT="${LLM_CONTEXT:-8192}"
LLM_GPU_LAYERS="${LLM_GPU_LAYERS:-auto}"

if [ -z "$LLM_MODEL" ]; then
  cat <<'EOF'
Set LLM_MODEL to either a local GGUF file or a Hugging Face llama.cpp model spec.

Examples:
  LLM_MODEL=models/local-llm/model.gguf \
    LLM_MMPROJ=models/local-llm/mmproj.gguf scripts/start-local-llm.sh

  LLM_MODEL=owner/model-gguf:Q4_K_M scripts/start-local-llm.sh

The model must support image input because Groot uses it to review generated images.
EOF
  exit 2
fi

if [[ "$LLM_MODEL" != /* ]] && [ -f "$REPO_ROOT/$LLM_MODEL" ]; then
  LLM_MODEL="$REPO_ROOT/$LLM_MODEL"
fi
if [ -n "$LLM_MMPROJ" ] && [[ "$LLM_MMPROJ" != /* ]]; then
  LLM_MMPROJ="$REPO_ROOT/$LLM_MMPROJ"
fi

if command -v llama-server >/dev/null 2>&1; then
  SERVER_BIN="$(command -v llama-server)"
  ARGS=()
elif command -v llama >/dev/null 2>&1; then
  SERVER_BIN="$(command -v llama)"
  ARGS=(serve)
else
  echo "llama-server (or the llama app) is not installed or not on PATH."
  exit 1
fi

ARGS+=(
  --host "$LLM_HOST"
  --port "$LLM_PORT"
  --alias "$LLM_ALIAS"
  --ctx-size "$LLM_CONTEXT"
  --n-gpu-layers "$LLM_GPU_LAYERS"
  --jinja
)

if [ -f "$LLM_MODEL" ]; then
  ARGS+=(-m "$LLM_MODEL")
else
  ARGS+=(-hf "$LLM_MODEL")
fi

if [ -n "$LLM_MMPROJ" ]; then
  if [ ! -f "$LLM_MMPROJ" ]; then
    echo "Multimodal projector not found: $LLM_MMPROJ"
    exit 1
  fi
  ARGS+=(--mmproj "$LLM_MMPROJ")
fi

echo "Starting local multimodal model as '$LLM_ALIAS' at http://$LLM_HOST:$LLM_PORT/v1"
exec "$SERVER_BIN" "${ARGS[@]}"
