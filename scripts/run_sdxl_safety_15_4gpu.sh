#!/usr/bin/env bash
set -euo pipefail

# Run the same matrix as run_sdxl_safety_15.sh, with prompt shards distributed
# across GPUs 0-3. Additional arguments are forwarded to the Python scheduler.
python scripts/run_sdxl_safety_15_4gpu.py "$@"
