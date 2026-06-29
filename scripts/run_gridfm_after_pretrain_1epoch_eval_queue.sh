#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_gridfm}"
PYTHON_BIN="${PYTHON_BIN:-/2024233240/miniconda3/envs/libero/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-${ROOT}/src}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
RAW_MAX_STEPS="${RAW_MAX_STEPS:-8679}"
RUN_TAG="${RUN_TAG:-gridfm_denoisy_seed${SEED}}"
PRETRAIN_SESSION="${PRETRAIN_SESSION:-gridfm_pretrain_20k}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/gridfm_1epoch_after_pretrain/logs}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/evaluate_results/gridfm_raw_libero_1epoch}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-/2024233240/if-wam_incoming/ifwam_data/text_embeds_cache}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/2024233240/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"

mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"
export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GIT_PYTHON_REFRESH=quiet
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/2024233240/if-wam/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

step_tag() {
  printf 'step_%06d.pt' "$1"
}

run_dir_for() {
  local cfg="$1" run_id="$2"
  printf '%s/runs/%s/%s' "${ROOT}" "${cfg}" "${run_id}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[gridfm-1epoch] missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[gridfm-1epoch] missing required directory: ${path}" >&2
    exit 1
  fi
}

wait_for_pretrain() {
  while tmux has-session -t "${PRETRAIN_SESSION}" 2>/dev/null; do
    echo "[gridfm-1epoch] waiting for tmux session ${PRETRAIN_SESSION} (poll=${WAIT_POLL_SECONDS}s)"
    sleep "${WAIT_POLL_SECONDS}"
  done
  echo "[gridfm-1epoch] pretrain session ${PRETRAIN_SESSION} is not running; checking checkpoints"
}

preflight() {
  require_file "${PYTHON_BIN}"
  require_file "${DATASET_STATS_PATH}"
  require_dir "${TEXT_CACHE_DIR}"
  for name in \
    libero_spatial_no_noops_lerobot \
    libero_object_no_noops_lerobot \
    libero_goal_no_noops_lerobot \
    libero_10_no_noops_lerobot
  do
    require_dir "/2024233240/sim_data/fastwam/data/libero_mujoco3.3.2/${name}"
  done
}

train_raw_1epoch() {
  local label="$1" cfg="$2" resume_path="$3"
  local run_id="${RUN_TAG}_${label}"
  local run_dir
  run_dir="$(run_dir_for "${cfg}" "${run_id}")"
  local log_file="${LOG_ROOT}/${run_id}_train.log"

  require_file "${resume_path}"
  if [[ -e "${run_dir}" ]]; then
    echo "[gridfm-1epoch] refusing to overwrite existing run dir: ${run_dir}" >&2
    exit 1
  fi

  echo "[gridfm-1epoch] raw train start label=${label} cfg=${cfg} max_steps=${RAW_MAX_STEPS} resume=${resume_path}"
  RUN_ID="${run_id}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
    "task=${cfg}" \
    "seed=${SEED}" \
    "max_steps=${RAW_MAX_STEPS}" \
    "resume=${resume_path}" \
    "wandb.enabled=${WANDB_ENABLED}" \
    "wandb.name=${run_id}" \
    "wandb.group=gridfm-raw-libero-1epoch" \
    2>&1 | tee "${log_file}"

  local final_ckpt="${run_dir}/checkpoints/weights/$(step_tag "${RAW_MAX_STEPS}")"
  require_file "${final_ckpt}"
  ln -sfn "${final_ckpt}" "${run_dir}/checkpoints/weights/final.pt"
  echo "[gridfm-1epoch] raw train done label=${label} final=${final_ckpt}"
}

eval_raw() {
  local label="$1" cfg="$2"
  local run_id="${RUN_TAG}_${label}"
  local ckpt
  ckpt="$(run_dir_for "${cfg}" "${run_id}")/checkpoints/weights/final.pt"
  local output_dir="${EVAL_ROOT}/${run_id}"
  local log_file="${EVAL_ROOT}/${run_id}_manager.log"

  require_file "${ckpt}"
  if [[ -e "${output_dir}" ]]; then
    echo "[gridfm-1epoch] refusing to overwrite existing eval dir: ${output_dir}" >&2
    exit 1
  fi

  echo "[gridfm-1epoch] eval start label=${label} ckpt=${ckpt} output=${output_dir} max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
  "${PYTHON_BIN}" experiments/libero/run_libero_manager.py \
    task=libero_uncond_2cam224_1e-4 \
    ckpt="${ckpt}" \
    MULTIRUN.num_gpus="${NUM_GPUS}" \
    MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU}" \
    EVALUATION.num_trials="${NUM_TRIALS}" \
    EVALUATION.output_dir="${output_dir}" \
    EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}" \
    +EVALUATION.text_embedding_cache_dir="${TEXT_CACHE_DIR}" \
    EVALUATION.num_inference_steps=10 \
    model.load_text_encoder=false \
    model.skip_dit_load_from_pretrain=true \
    model.action_dit_pretrained_path=null \
    2>&1 | tee "${log_file}"

  require_file "${output_dir}/summary.csv"
  echo "[gridfm-1epoch] eval done label=${label} summary=${output_dir}/summary.csv"
}

train_then_eval() {
  local label="$1" cfg="$2" resume_path="$3"
  train_raw_1epoch "${label}" "${cfg}" "${resume_path}"
  eval_raw "${label}" "${cfg}"
}

preflight
wait_for_pretrain

libero_pretrain_ckpt="$(run_dir_for "ifwam_libero_gridfm_pretrain_20k" "${RUN_TAG}_ifwam_libero_gridfm_pretrain_20k")/checkpoints/weights/step_020000.pt"
mixed_pretrain_ckpt="$(run_dir_for "ifwam_mixed_gridfm_pretrain_20k" "${RUN_TAG}_ifwam_mixed_gridfm_pretrain_20k")/checkpoints/weights/step_020000.pt"

train_then_eval "ifwam_libero_gridfm_raw_grounding_1epoch" "ifwam_libero_gridfm_raw_grounding_1epoch" "${libero_pretrain_ckpt}"
train_then_eval "ifwam_mixed_gridfm_raw_grounding_1epoch" "ifwam_mixed_gridfm_raw_grounding_1epoch" "${mixed_pretrain_ckpt}"

echo "[gridfm-1epoch] complete RUN_TAG=${RUN_TAG}"
echo "  libero gridfm summary: ${EVAL_ROOT}/${RUN_TAG}_ifwam_libero_gridfm_raw_grounding_1epoch/summary.csv"
echo "  mixed gridfm summary: ${EVAL_ROOT}/${RUN_TAG}_ifwam_mixed_gridfm_raw_grounding_1epoch/summary.csv"
