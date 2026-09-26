#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start-local-llm.sh" \
  --config "$SCRIPT_DIR/../configs/attacks/daca.yaml" \
  --section daca_llm \
  "$@"
