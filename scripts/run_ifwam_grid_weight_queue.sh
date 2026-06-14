#!/usr/bin/env bash
set -euo pipefail

ROOT="/2024233240/if-wam"
cd "${ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
COMPARE_STEPS="${COMPARE_STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
L_GRID="${L_GRID:-0.5}"
QUEUE_GROUP="${QUEUE_GROUP:-ifwam-grid-weight-compare}"
QUEUE_ID="${QUEUE_ID:-grid_weight_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/ifwam_grid_compare_logs/${QUEUE_ID}}"
mkdir -p "${LOG_DIR}"

run_grid() {
  local tag="$1"
  local manifest="$2"
  local run_id="${QUEUE_ID}_${tag}_grid_${L_GRID}"
  local run_log="${LOG_DIR}/${tag}_grid_${L_GRID}.log"

  printf '[grid-weight] start dataset=%s lambda_grid=%s manifest=%s\n' "${tag}" "${L_GRID}" "${manifest}"
  RUN_ID="${run_id}" bash scripts/train_ifwam_mixed_wandb.sh "${NPROC_PER_NODE}" \
    max_steps="${COMPARE_STEPS}" log_every="${LOG_EVERY}" gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    save_every=0 save_weights_every=0 save_state_every=0 save_final_state=false eval_every=0 \
    wandb.group="${QUEUE_GROUP}" wandb.name="${tag}_grid_${L_GRID}_${QUEUE_ID}" wandb.core_metrics_only=true \
    data.train.manifest_path="${manifest}" \
    ifwam.ifwam.mot.return_mid_states=true \
    ifwam.ifwam.process_flow_readout.enabled=true \
    ifwam.ifwam.process_flow_readout.use_video_readout=false \
    ifwam.ifwam.process_flow_readout.use_action_readout=false \
    ifwam.ifwam.process_flow_readout.use_grid_readout=true \
    ifwam.ifwam.scoring_head.enabled=false \
    ifwam.ifwam.losses.lambda_vflow=0.0 \
    ifwam.ifwam.losses.lambda_aflow=0.0 \
    ifwam.ifwam.losses.lambda_gridflow="${L_GRID}" \
    2>&1 | tee "${run_log}"
  printf '[grid-weight] done dataset=%s lambda_grid=%s\n' "${tag}" "${L_GRID}"
}

unset PYTORCH_CUDA_ALLOC_CONF
run_grid mixed /2024233240/ifwam_data/manifests/train_mixed_grid_rgb.jsonl
run_grid libero /2024233240/ifwam_data/manifests/train_libero_grid_rgb.jsonl
printf '[grid-weight] all done id=%s lambda_grid=%s\n' "${QUEUE_ID}" "${L_GRID}"
