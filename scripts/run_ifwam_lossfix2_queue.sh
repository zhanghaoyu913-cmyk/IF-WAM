#!/usr/bin/env bash
set -euo pipefail
ROOT="/2024233240/if-wam"
cd "${ROOT}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-1000}"
SAVE_WEIGHTS_EVERY="${SAVE_WEIGHTS_EVERY:-500}"
SAVE_STATE_EVERY="${SAVE_STATE_EVERY:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
QUEUE_GROUP="${QUEUE_GROUP:-ifwam-lossfix2}"
QUEUE_ID="${QUEUE_ID:-lossfix2_$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/ifwam_ablation_queue_logs/${QUEUE_ID}}"
mkdir -p "${LOG_DIR}"

# Targeted follow-up after lossfix_2026-06-11_12-05-00.
# name lambda_vflow lambda_aflow lambda_gridflow
EXPERIMENTS=(
  "only_aflow_fixed 0.0 0.01 0.0"
  "only_grid_direction 0.0 0.0 0.01"
  "vflow_aflow_fixed 0.02 0.01 0.0"
  "full_fixed 0.02 0.01 0.01"
)

printf '[queue] id=%s nproc=%s steps=%s save_state_every=%s log_dir=%s\n' "${QUEUE_ID}" "${NPROC_PER_NODE}" "${ABLATION_STEPS}" "${SAVE_STATE_EVERY}" "${LOG_DIR}"
for spec in "${EXPERIMENTS[@]}"; do
  read -r EXP_NAME L_VFLOW L_AFLOW L_GRIDFLOW <<< "${spec}"
  RUN_ID="${QUEUE_ID}_${EXP_NAME}"
  RUN_LOG="${LOG_DIR}/${EXP_NAME}.log"
  printf '[queue] start exp=%s vflow=%s aflow=%s gridflow=%s\n' "${EXP_NAME}" "${L_VFLOW}" "${L_AFLOW}" "${L_GRIDFLOW}"
  RUN_ID="${RUN_ID}" bash scripts/train_ifwam_mixed_wandb.sh "${NPROC_PER_NODE}" \
    max_steps="${ABLATION_STEPS}" log_every="${LOG_EVERY}" \
    save_every=0 save_weights_every="${SAVE_WEIGHTS_EVERY}" save_state_every="${SAVE_STATE_EVERY}" save_final_state=false \
    wandb.group="${QUEUE_GROUP}" wandb.name="${EXP_NAME}_${QUEUE_ID}" \
    ifwam.ifwam.losses.lambda_vflow="${L_VFLOW}" \
    ifwam.ifwam.losses.lambda_aflow="${L_AFLOW}" \
    ifwam.ifwam.losses.lambda_gridflow="${L_GRIDFLOW}" \
    2>&1 | tee "${RUN_LOG}"
  printf '[queue] done exp=%s\n' "${EXP_NAME}"
done
printf '[queue] all done id=%s\n' "${QUEUE_ID}"
