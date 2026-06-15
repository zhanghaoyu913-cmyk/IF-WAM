#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_incoming/if-wam}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-${ROOT}/src}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODE="${MODE:-debug}" # debug or full
SEEDS="${SEEDS:-0}"
DEBUG_STAGE_A_STEPS="${DEBUG_STAGE_A_STEPS:-2}"
DEBUG_STAGE_B_STEPS="${DEBUG_STAGE_B_STEPS:-1}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/manifest_grid_ablation_queue/$(date +%Y%m%d_%H%M%S)}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"
export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/2024233240/if-wam/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

experiments=(
  "ifwam_libero_manifest_action_18k_g2k ifwam_libero_manifest_action_18k_g2k_stage_a ifwam_libero_manifest_action_18k_g2k_stage_b"
  "ifwam_libero_manifest_grid_18k_g2k ifwam_libero_manifest_grid_18k_g2k_stage_a ifwam_libero_manifest_grid_18k_g2k_stage_b"
  "ifwam_mixed_manifest_action_18k_g2k ifwam_mixed_manifest_action_18k_g2k_stage_a ifwam_mixed_manifest_action_18k_g2k_stage_b"
  "ifwam_mixed_manifest_grid_18k_g2k ifwam_mixed_manifest_grid_18k_g2k_stage_a ifwam_mixed_manifest_grid_18k_g2k_stage_b"
)

step_tag() {
  printf 'step_%06d.pt' "$1"
}

run_stage() {
  local exp="$1" stage="$2" cfg="$3" seed="$4" max_steps_override="$5" resume_path="$6"
  local run_id="${exp}_seed${seed}_${stage}"
  local log_file="${LOG_ROOT}/${run_id}.log"
  local -a args=(
    "task=${cfg}"
    "seed=${seed}"
    "wandb.enabled=${WANDB_ENABLED}"
    "wandb.name=${run_id}"
    "wandb.group=ifwam-manifest-grid-ablation"
  )
  if [[ -n "${max_steps_override}" ]]; then
    args+=("max_steps=${max_steps_override}")
  fi
  if [[ -n "${resume_path}" ]]; then
    args+=("resume=${resume_path}")
  fi
  echo "[queue] start ${run_id} cfg=${cfg} resume=${resume_path:-none} log=${log_file}"
  RUN_ID="${run_id}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${args[@]}" 2>&1 | tee "${log_file}"
  echo "[queue] done ${run_id}"
}

if [[ "${MODE}" == "debug" ]]; then
  stage_a_steps="${DEBUG_STAGE_A_STEPS}"
  stage_b_steps="${DEBUG_STAGE_B_STEPS}"
else
  stage_a_steps=""
  stage_b_steps=""
fi

for seed in ${SEEDS}; do
  for spec in "${experiments[@]}"; do
    read -r exp cfg_a cfg_b <<< "${spec}"
    run_stage "${exp}" "stage_a" "${cfg_a}" "${seed}" "${stage_a_steps}" ""
    if [[ "${MODE}" == "debug" ]]; then
      ckpt_step="${DEBUG_STAGE_A_STEPS}"
    else
      ckpt_step=18000
    fi
    ckpt="${ROOT}/runs/${cfg_a}/${exp}_seed${seed}_stage_a/checkpoints/weights/$(step_tag "${ckpt_step}")"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[queue] missing Stage A checkpoint: ${ckpt}" >&2
      exit 1
    fi
    run_stage "${exp}" "stage_b" "${cfg_b}" "${seed}" "${stage_b_steps}" "${ckpt}"
  done
done

echo "[queue] all training stages completed mode=${MODE} seeds=${SEEDS} log_root=${LOG_ROOT}"
