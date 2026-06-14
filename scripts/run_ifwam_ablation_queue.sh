#!/usr/bin/env bash
set -euo pipefail

ROOT="/2024233240/if-wam"
cd "${ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-1000}"
SAVE_WEIGHTS_EVERY="${SAVE_WEIGHTS_EVERY:-500}"
SAVE_STATE_EVERY="${SAVE_STATE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
QUEUE_GROUP="${QUEUE_GROUP:-ifwam-ablation-v1}"
QUEUE_ID="${QUEUE_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/ifwam_ablation_queue_logs/${QUEUE_ID}}"
mkdir -p "${LOG_DIR}"

# name lambda_vflow lambda_aflow lambda_gridflow
EXPERIMENTS=(
  "baseline_no_flow 0.0 0.0 0.0"
  "only_gridflow 0.0 0.0 0.01"
  "only_vflow_low 0.02 0.0 0.0"
  "vflow_aflow_low 0.02 0.01 0.0"
  "full_safe 0.02 0.01 0.01"
)

printf '[queue] id=%s nproc=%s steps=%s log_dir=%s\n' "${QUEUE_ID}" "${NPROC_PER_NODE}" "${ABLATION_STEPS}" "${LOG_DIR}"

for spec in "${EXPERIMENTS[@]}"; do
  read -r EXP_NAME L_VFLOW L_AFLOW L_GRIDFLOW <<< "${spec}"
  RUN_ID="${QUEUE_ID}_${EXP_NAME}"
  RUN_LOG="${LOG_DIR}/${EXP_NAME}.log"
  printf '[queue] start exp=%s vflow=%s aflow=%s gridflow=%s log=%s\n' "${EXP_NAME}" "${L_VFLOW}" "${L_AFLOW}" "${L_GRIDFLOW}" "${RUN_LOG}"

  RUN_ID="${RUN_ID}" bash scripts/train_ifwam_mixed_wandb.sh "${NPROC_PER_NODE}" \
    max_steps="${ABLATION_STEPS}" \
    log_every="${LOG_EVERY}" \
    save_every=0 \
    save_weights_every="${SAVE_WEIGHTS_EVERY}" \
    save_state_every="${SAVE_STATE_EVERY}" \
    wandb.group="${QUEUE_GROUP}" \
    wandb.name="${EXP_NAME}_${QUEUE_ID}" \
    ifwam.ifwam.losses.lambda_vflow="${L_VFLOW}" \
    ifwam.ifwam.losses.lambda_aflow="${L_AFLOW}" \
    ifwam.ifwam.losses.lambda_gridflow="${L_GRIDFLOW}" \
    2>&1 | tee "${RUN_LOG}"

  printf '[queue] done exp=%s\n' "${EXP_NAME}"
done

printf '[queue] all done id=%s\n' "${QUEUE_ID}"
