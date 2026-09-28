#!/usr/bin/env bash
set -euo pipefail

# Evaluate PGJ with fresh candidates against SAFREE, Latent Guard, and TraSCE.
# Additional arguments are forwarded to the Python scheduler.
python scripts/run_pgj_defenses_4gpu.py "$@"
