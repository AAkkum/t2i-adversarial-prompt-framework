#!/usr/bin/env bash
set -euo pipefail

# Evaluate Latent Guard against the five paper-evaluation attacks on SD 1.4
# and SD 3.5 Medium. Additional arguments go to the scheduler.
python scripts/run_latent_guard_all_4gpu.py "$@"
