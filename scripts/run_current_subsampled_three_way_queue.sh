#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_incoming/if-wam}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-${ROOT}/src}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/current_subsampled_three_way/logs}"
RUN_TAG="${RUN_TAG:-current_subsampled_seed${SEED}}"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"
export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/2024233240/if-wam/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

step_tag() {
  printf 'step_%06d.pt' "$1"
}

run_dir_for() {
  local cfg="$1" run_id="$2"
  printf '%s/runs/%s/%s' "${ROOT}" "${cfg}" "${run_id}"
}

prepare_run_dir() {
  local dir="$1"
  if [[ "${dir}" != "${ROOT}/runs/"* ]]; then
    echo "[queue] refusing to clean unexpected run dir: ${dir}" >&2
    exit 1
  fi
  rm -rf "${dir}"
  mkdir -p "$(dirname "${dir}")"
}

run_stage() {
  local label="$1" cfg="$2" max_steps="$3" resume_path="$4"
  local run_id="${RUN_TAG}_${label}"
  local run_dir
  run_dir="$(run_dir_for "${cfg}" "${run_id}")"
  local log_file="${LOG_ROOT}/${run_id}.log"

  prepare_run_dir "${run_dir}"

  local -a args=(
    "task=${cfg}"
    "seed=${SEED}"
    "max_steps=${max_steps}"
    "resume=null"
    "wandb.enabled=${WANDB_ENABLED}"
    "wandb.name=${run_id}"
    "wandb.group=current-subsampled-three-way"
  )
  if [[ -n "${resume_path}" ]]; then
    args+=("resume=${resume_path}")
  fi

  echo "[queue] start ${label} cfg=${cfg} max_steps=${max_steps} resume=${resume_path:-none} run_dir=${run_dir}"
  RUN_ID="${run_id}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${args[@]}" 2>&1 | tee "${log_file}"

  local final_ckpt="${run_dir}/checkpoints/weights/$(step_tag "${max_steps}")"
  if [[ ! -f "${final_ckpt}" ]]; then
    echo "[queue] missing final checkpoint: ${final_ckpt}" >&2
    exit 1
  fi
  find "${run_dir}/checkpoints/weights" -type f ! -name "$(basename "${final_ckpt}")" -delete
  if [[ -d "${run_dir}/checkpoints/state" ]]; then
    find "${run_dir}/checkpoints/state" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
  ln -sfn "${final_ckpt}" "${run_dir}/checkpoints/weights/final.pt"
  echo "[queue] done ${label} final=${final_ckpt}"
}

# 1. Fast-WAM architecture + current subsampled LIBERO manifest, from base init.
run_stage "fastwam_libero_manifest_20k" "fastwam_libero_manifest_20k" 20000 ""

# 2. IF-WAM + current subsampled LIBERO + LIBERO grid flow.
run_stage "ifwam_libero_grid_pretrain_20k" "ifwam_libero_grid_pretrain_20k" 20000 ""

# 3. IF-WAM + current subsampled mixed data + mixed grid flow, then LIBERO action grounding.
run_stage "ifwam_mixed_grid_pretrain_20k" "ifwam_mixed_grid_pretrain_20k" 20000 ""
mixed_pretrain_dir="$(run_dir_for "ifwam_mixed_grid_pretrain_20k" "${RUN_TAG}_ifwam_mixed_grid_pretrain_20k")"
mixed_pretrain_ckpt="${mixed_pretrain_dir}/checkpoints/weights/$(step_tag 20000)"
run_stage "ifwam_mixed_grid_grounding_2k" "ifwam_mixed_grid_grounding_2k" 2000 "${mixed_pretrain_ckpt}"

echo "[queue] complete current subsampled three-way training seed=${SEED}"
echo "[queue] final checkpoints:"
echo "  fastwam_libero: $(run_dir_for "fastwam_libero_manifest_20k" "${RUN_TAG}_fastwam_libero_manifest_20k")/checkpoints/weights/final.pt"
echo "  ifwam_libero_grid: $(run_dir_for "ifwam_libero_grid_pretrain_20k" "${RUN_TAG}_ifwam_libero_grid_pretrain_20k")/checkpoints/weights/final.pt"
echo "  ifwam_mixed_grid: $(run_dir_for "ifwam_mixed_grid_grounding_2k" "${RUN_TAG}_ifwam_mixed_grid_grounding_2k")/checkpoints/weights/final.pt"
