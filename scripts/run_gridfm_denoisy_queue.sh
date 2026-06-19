#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_gridfm}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-${ROOT}/src}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/gridfm_denoisy_queue/logs}"
RUN_TAG="${RUN_TAG:-gridfm_denoisy_seed${SEED}}"
WAIT_FOR_TMUX_SESSIONS="${WAIT_FOR_TMUX_SESSIONS:-corrected_current_subsampled current_subsampled_train fastwam_singleview_eval_v2 libero_test_v3}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"

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
    echo "[gridfm-queue] refusing to clean unexpected run dir: ${dir}" >&2
    exit 1
  fi
  rm -rf "${dir}"
  mkdir -p "$(dirname "${dir}")"
}

wait_for_tmux_sessions() {
  local -a sessions=()
  read -r -a sessions <<< "${WAIT_FOR_TMUX_SESSIONS}"
  if (( ${#sessions[@]} == 0 )); then
    return 0
  fi

  while true; do
    local -a alive=()
    local session
    for session in "${sessions[@]}"; do
      if tmux has-session -t "${session}" 2>/dev/null; then
        alive+=("${session}")
      fi
    done
    if (( ${#alive[@]} == 0 )); then
      echo "[gridfm-queue] no blocking tmux sessions remain; starting GridFM queue"
      return 0
    fi
    echo "[gridfm-queue] waiting for tmux sessions: ${alive[*]} (poll=${WAIT_POLL_SECONDS}s)"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

keep_only_final_checkpoint() {
  local run_dir="$1" max_steps="$2"
  local final_ckpt="${run_dir}/checkpoints/weights/$(step_tag "${max_steps}")"
  if [[ ! -f "${final_ckpt}" ]]; then
    echo "[gridfm-queue] missing final checkpoint: ${final_ckpt}" >&2
    exit 1
  fi
  find "${run_dir}/checkpoints/weights" -type f ! -name "$(basename "${final_ckpt}")" -delete
  if [[ -d "${run_dir}/checkpoints/state" ]]; then
    find "${run_dir}/checkpoints/state" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
  ln -sfn "${final_ckpt}" "${run_dir}/checkpoints/weights/final.pt"
  echo "${final_ckpt}"
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
    "wandb.group=gridfm-denoisy-same-setting"
  )
  if [[ -n "${resume_path}" ]]; then
    args+=("resume=${resume_path}")
  fi

  echo "[gridfm-queue] start ${label} cfg=${cfg} max_steps=${max_steps} resume=${resume_path:-none} run_dir=${run_dir}"
  RUN_ID="${run_id}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${args[@]}" 2>&1 | tee "${log_file}"

  local final_ckpt
  final_ckpt="$(keep_only_final_checkpoint "${run_dir}" "${max_steps}")"
  echo "[gridfm-queue] done ${label} final=${final_ckpt}"
}

wait_for_tmux_sessions

# Same current-subsampled setting as the active three-way comparison:
# 4 GPUs, global batch 32, seed 0 by default, 20k grid pretrain + 2k LIBERO action grounding.
run_stage "ifwam_libero_gridfm_pretrain_20k" "ifwam_libero_gridfm_pretrain_20k" 20000 ""
libero_pretrain_dir="$(run_dir_for "ifwam_libero_gridfm_pretrain_20k" "${RUN_TAG}_ifwam_libero_gridfm_pretrain_20k")"
libero_pretrain_ckpt="${libero_pretrain_dir}/checkpoints/weights/$(step_tag 20000)"
run_stage "ifwam_libero_gridfm_raw_grounding_2k" "ifwam_libero_gridfm_raw_grounding_2k" 2000 "${libero_pretrain_ckpt}"

run_stage "ifwam_mixed_gridfm_pretrain_20k" "ifwam_mixed_gridfm_pretrain_20k" 20000 ""
mixed_pretrain_dir="$(run_dir_for "ifwam_mixed_gridfm_pretrain_20k" "${RUN_TAG}_ifwam_mixed_gridfm_pretrain_20k")"
mixed_pretrain_ckpt="${mixed_pretrain_dir}/checkpoints/weights/$(step_tag 20000)"
run_stage "ifwam_mixed_gridfm_raw_grounding_2k" "ifwam_mixed_gridfm_raw_grounding_2k" 2000 "${mixed_pretrain_ckpt}"

echo "[gridfm-queue] start raw/full LIBERO eval"
RUN_TAG="${RUN_TAG}" bash scripts/run_gridfm_raw_libero_eval_queue.sh

echo "[gridfm-queue] complete GridFM denoisy training seed=${SEED}"
echo "[gridfm-queue] final checkpoints:"
echo "  ifwam_libero_gridfm: $(run_dir_for "ifwam_libero_gridfm_raw_grounding_2k" "${RUN_TAG}_ifwam_libero_gridfm_raw_grounding_2k")/checkpoints/weights/final.pt"
echo "  ifwam_mixed_gridfm: $(run_dir_for "ifwam_mixed_gridfm_raw_grounding_2k" "${RUN_TAG}_ifwam_mixed_gridfm_raw_grounding_2k")/checkpoints/weights/final.pt"
