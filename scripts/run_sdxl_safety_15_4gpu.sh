#!/usr/bin/env bash
set -euo pipefail

# Run the sharded safety matrix across GPUs 0-3. The legacy script name is kept
# for compatibility; the scheduler also includes matched SD 1.4 none/TraSCE cases.
# Additional arguments are forwarded to the Python scheduler.
python scripts/run_sdxl_safety_15_4gpu.py "$@"
