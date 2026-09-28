#!/usr/bin/env bash
set -euo pipefail

# Generate fresh PGJ and Ring-A-Bell candidates, then reuse the same candidates
# across defenses. Additional arguments are forwarded to the Python scheduler.
python scripts/run_pgj_ring_all_defenses_4gpu.py "$@"
