#!/usr/bin/env bash
set -euo pipefail
ROOT="/2024233240/if-wam"
cd "${ROOT}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
COMPARE_STEPS="${COMPARE_STEPS:-1000}"
SAVE_WEIGHTS_EVERY="${SAVE_WEIGHTS_EVERY:-0}"
SAVE_STATE_EVERY="${SAVE_STATE_EVERY:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
QUEUE_GROUP="${QUEUE_GROUP:-ifwam-grid-only-compare}"
QUEUE_ID="${QUEUE_ID:-grid_compare_$(date +%Y-%m-%d_%H-%M-%S)}"
MANIFEST="${MANIFEST:-/2024233240/ifwam_data/manifests/train_mixed_grid_rgb.jsonl}"
L_GRID="${L_GRID:-0.03}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/ifwam_grid_compare_logs/${QUEUE_ID}}"
mkdir -p "${LOG_DIR}"

run_one() {
  local exp_name="$1" grid_enabled="$2" l_grid="$3"
  local run_id="${QUEUE_ID}_${exp_name}"
  local run_log="${LOG_DIR}/${exp_name}.log"
  printf '[grid-compare] start exp=%s grid_enabled=%s lambda_grid=%s manifest=%s\n' "${exp_name}" "${grid_enabled}" "${l_grid}" "${MANIFEST}"
  if [[ "${grid_enabled}" == "true" ]]; then
    RUN_ID="${run_id}" bash scripts/train_ifwam_mixed_wandb.sh "${NPROC_PER_NODE}" \
      max_steps="${COMPARE_STEPS}" log_every="${LOG_EVERY}" gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}" \
      save_every=0 save_weights_every="${SAVE_WEIGHTS_EVERY}" save_state_every="${SAVE_STATE_EVERY}" save_final_state=false eval_every=0 \
      wandb.group="${QUEUE_GROUP}" wandb.name="${exp_name}_${QUEUE_ID}" wandb.core_metrics_only=true \
      data.train.manifest_path="${MANIFEST}" \
      ifwam.ifwam.mot.return_mid_states=true \
      ifwam.ifwam.process_flow_readout.enabled=true \
      ifwam.ifwam.process_flow_readout.use_video_readout=false \
      ifwam.ifwam.process_flow_readout.use_action_readout=false \
      ifwam.ifwam.process_flow_readout.use_grid_readout=true \
      ifwam.ifwam.scoring_head.enabled=false \
      ifwam.ifwam.losses.lambda_vflow=0.0 \
      ifwam.ifwam.losses.lambda_aflow=0.0 \
      ifwam.ifwam.losses.lambda_gridflow="${l_grid}" \
      2>&1 | tee "${run_log}"
  else
    RUN_ID="${run_id}" bash scripts/train_ifwam_mixed_wandb.sh "${NPROC_PER_NODE}" \
      max_steps="${COMPARE_STEPS}" log_every="${LOG_EVERY}" gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}" \
      save_every=0 save_weights_every="${SAVE_WEIGHTS_EVERY}" save_state_every="${SAVE_STATE_EVERY}" save_final_state=false eval_every=0 \
      wandb.group="${QUEUE_GROUP}" wandb.name="${exp_name}_${QUEUE_ID}" wandb.core_metrics_only=true \
      data.train.manifest_path="${MANIFEST}" \
      ifwam.ifwam.mot.return_mid_states=false \
      ifwam.ifwam.process_flow_readout.enabled=false \
      ifwam.ifwam.process_flow_readout.use_video_readout=false \
      ifwam.ifwam.process_flow_readout.use_action_readout=false \
      ifwam.ifwam.process_flow_readout.use_grid_readout=false \
      ifwam.ifwam.scoring_head.enabled=false \
      ifwam.ifwam.losses.lambda_vflow=0.0 \
      ifwam.ifwam.losses.lambda_aflow=0.0 \
      ifwam.ifwam.losses.lambda_gridflow=0.0 \
      2>&1 | tee "${run_log}"
  fi
  printf '[grid-compare] done exp=%s\n' "${exp_name}"
}

printf '[grid-compare] id=%s steps=%s nproc=%s l_grid=%s log_dir=%s\n' "${QUEUE_ID}" "${COMPARE_STEPS}" "${NPROC_PER_NODE}" "${L_GRID}" "${LOG_DIR}"
run_one baseline_no_grid false 0.0
run_one grid_only true "${L_GRID}"
printf '[grid-compare] all done id=%s\n' "${QUEUE_ID}"
