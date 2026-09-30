#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

cd "$REPO_ROOT"
# Keep one physical GPU visible by default. Users can choose another one, for
# example: CUDA_VISIBLE_DEVICES=2 scripts/start-local-llm.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$PYTHON" -m t2i_framework.core.local_llm_server "$@"
