#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-gridfm_one_way_direction_presence}"
SEED="${SEED:-0}"
CHECKPOINT="${CHECKPOINT:-null}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/gridflow_ablation/${CONFIG}_seed${SEED}}"
DRY_RUN="${DRY_RUN:-1}"
MAX_STEPS="${MAX_STEPS:-20000}"

CMD=(
  python scripts/train.py
  task="${CONFIG}"
  seed="${SEED}"
  resume="${CHECKPOINT}"
  max_steps="${MAX_STEPS}"
  output_dir="${OUTPUT_DIR}"
)

printf 'Command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

exec "${CMD[@]}"
