#!/usr/bin/env bash
set -euo pipefail
ROOT="/2024233240/if-wam"
cd "${ROOT}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
COMPARE_STEPS="${COMPARE_STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_WEIGHTS_EVERY="${SAVE_WEIGHTS_EVERY:-0}"
SAVE_STATE_EVERY="${SAVE_STATE_EVERY:-0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
L_GRID="${L_GRID:-0.03}"
QUEUE_GROUP="${QUEUE_GROUP:-ifwam-grid-compare-suite-ga16}"
QUEUE_ID="${QUEUE_ID:-grid_suite_ga16_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/ifwam_grid_compare_logs/${QUEUE_ID}}"
mkdir -p "${LOG_DIR}"

run_pair() {
  local tag="$1" manifest="$2"
  printf '[suite] dataset=%s manifest=%s\n' "${tag}" "${manifest}"
  MANIFEST="${manifest}" \
  COMPARE_STEPS="${COMPARE_STEPS}" LOG_EVERY="${LOG_EVERY}" \
  SAVE_WEIGHTS_EVERY="${SAVE_WEIGHTS_EVERY}" SAVE_STATE_EVERY="${SAVE_STATE_EVERY}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
  L_GRID="${L_GRID}" QUEUE_GROUP="${QUEUE_GROUP}" QUEUE_ID="${QUEUE_ID}_${tag}" LOG_DIR="${LOG_DIR}/${tag}" \
  bash scripts/run_ifwam_grid_compare_queue.sh
}

printf '[suite] id=%s steps=%s ga=%s l_grid=%s\n' "${QUEUE_ID}" "${COMPARE_STEPS}" "${GRADIENT_ACCUMULATION_STEPS}" "${L_GRID}"
run_pair mixed /2024233240/ifwam_data/manifests/train_mixed_grid_rgb.jsonl
run_pair libero /2024233240/ifwam_data/manifests/train_libero_grid_rgb.jsonl
printf '[suite] all done id=%s\n' "${QUEUE_ID}"
