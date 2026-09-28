#!/usr/bin/env bash
set -euo pipefail

# Evaluate PGJ with fresh candidates against SAFREE and TraSCE, plus baselines.
# Additional arguments are forwarded to the Python scheduler.
python scripts/run_pgj_defenses_4gpu.py "$@"
