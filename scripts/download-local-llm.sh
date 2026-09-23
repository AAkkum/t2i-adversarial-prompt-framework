#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_REPO="${1:-${MODEL_REPO:-}}"
MODEL_PATTERNS="${2:-${MODEL_PATTERNS:-}}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/local-llm}"

if [ -z "$MODEL_REPO" ] || [ -z "$MODEL_PATTERNS" ]; then
  cat <<'EOF'
Usage:
  scripts/download-local-llm.sh REPOSITORY FILE_PATTERN[,FILE_PATTERN...]

Example:
  scripts/download-local-llm.sh owner/model-gguf '*Q4_K_M.gguf,*mmproj*.gguf'

The selected files are stored in models/local-llm by default. Set MODEL_DIR to
use another directory. For gated repositories, authenticate first with:
  hf auth login
EOF
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

if ! "$PYTHON_BIN" -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "huggingface_hub is missing. Install the model dependencies first:"
  echo '  pip install -e ".[models]"'
  exit 1
fi

mkdir -p "$MODEL_DIR"
echo "Repository:  $MODEL_REPO"
echo "Patterns:    $MODEL_PATTERNS"
echo "Destination: $MODEL_DIR"

MODEL_REPO="$MODEL_REPO" MODEL_PATTERNS="$MODEL_PATTERNS" MODEL_DIR="$MODEL_DIR" \
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

patterns = [item.strip() for item in os.environ["MODEL_PATTERNS"].split(",") if item.strip()]
target = Path(os.environ["MODEL_DIR"]).expanduser().resolve()
snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    local_dir=target,
    allow_patterns=patterns,
)

print("Downloaded GGUF files:")
for path in sorted(target.rglob("*.gguf")):
    print(f"  {path}")
PY
