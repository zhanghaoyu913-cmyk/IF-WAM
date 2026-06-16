#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_incoming/if-wam}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-${ROOT}/src}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEED="${SEED:-0}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/fastwam_comparison_queue/$(date +%Y%m%d_%H%M%S)}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"
export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/2024233240/if-wam/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

step_tag() {
  printf 'step_%06d.pt' "$1"
}

run_stage() {
  local exp="$1" stage="$2" cfg="$3" max_steps="$4" resume_path="$5"
  local run_id="${exp}_seed${SEED}_${stage}"
  local log_file="${LOG_ROOT}/${run_id}.log"
  local -a args=(
    "task=${cfg}"
    "seed=${SEED}"
    "max_steps=${max_steps}"
    "wandb.enabled=${WANDB_ENABLED}"
    "wandb.name=${run_id}"
    "wandb.group=ifwam-fastwam-comparison"
  )
  if [[ -n "${resume_path}" ]]; then
    args+=("resume=${resume_path}")
  fi
  echo "[queue] start ${run_id} cfg=${cfg} resume=${resume_path:-none} log=${log_file}"
  RUN_ID="${run_id}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${args[@]}" 2>&1 | tee "${log_file}"
  echo "[queue] done ${run_id}"
}

experiments=(
  "ifwam_libero_grid ifwam_libero_grid_pretrain_20k ifwam_libero_grid_grounding_2k"
  "ifwam_mixed_grid ifwam_mixed_grid_pretrain_20k ifwam_mixed_grid_grounding_2k"
)

for spec in "${experiments[@]}"; do
  read -r exp pre_cfg ground_cfg <<< "${spec}"
  run_stage "${exp}" "pretrain" "${pre_cfg}" 20000 ""
  ckpt="${ROOT}/runs/${pre_cfg}/${exp}_seed${SEED}_pretrain/checkpoints/weights/$(step_tag 20000)"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[queue] missing pretrain checkpoint: ${ckpt}" >&2
    exit 1
  fi
  run_stage "${exp}" "grounding" "${ground_cfg}" 2000 "${ckpt}"
done

echo "[queue] all IF-WAM Fast-WAM comparison training stages completed seed=${SEED} log_root=${LOG_ROOT}"
