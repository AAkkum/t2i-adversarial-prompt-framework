#!/usr/bin/env bash
set -euo pipefail

# SDXL safety matrix. CharacterFilter and TraSCE are intentionally excluded.
# For a prompt-sharded multi-GPU run, use run_sdxl_safety_15_4gpu.sh.

DATASET="${1:-data/datasets/safety_nonsexual/safety_nonsexual_100.csv}"
RUN_GROUP="${RUN_GROUP:-$(date +%Y%m%d_%H%M%S)_sdxl_safety_15}"
OUTPUT_ROOT="results/matrices/${RUN_GROUP}"

run_case() {
  local number="$1"
  local attack="$2"
  local defense="$3"
  local max_candidates="$4"
  shift 4

  echo "[$number/15] attack=$attack defense=$defense max_candidates=$max_candidates"
  python main.py \
    --model diffusers \
    --model-config configs/models/sdxl.yaml \
    --attack "$attack" \
    --defense "$defense" \
    --prompt-file "$DATASET" \
    --max-candidates "$max_candidates" \
    --out "$OUTPUT_ROOT/${number}_${attack}_${defense}" \
    "$@"
}

# Original-prompt controls.
run_case 01 identity none 1
run_case 02 identity latent_guard_lite 1 \
  --defense-config configs/defenses/latent_guard_safety_nonsexual.yaml
run_case 03 identity safree 1

# Each attack without a defense, followed by SAFREE and LatentGuard.
run_case 04 pgj none 1
run_case 05 pgj safree 1
run_case 06 pgj latent_guard_lite 1 \
  --defense-config configs/defenses/latent_guard_safety_nonsexual.yaml

run_case 07 daca none 10
run_case 08 daca safree 10
run_case 09 daca latent_guard_lite 10 \
  --defense-config configs/defenses/latent_guard_safety_nonsexual.yaml

run_case 10 groot none 3
run_case 11 groot safree 3
run_case 12 groot latent_guard_lite 3 \
  --defense-config configs/defenses/latent_guard_safety_nonsexual.yaml

run_case 13 ring_a_bell none 1
run_case 14 ring_a_bell safree 1
run_case 15 ring_a_bell latent_guard_lite 1 \
  --defense-config configs/defenses/latent_guard_safety_nonsexual.yaml

echo "Completed 15 runs in $OUTPUT_ROOT"
