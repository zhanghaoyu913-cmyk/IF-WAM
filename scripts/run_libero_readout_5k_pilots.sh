#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_gridfm}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-${ROOT}/src}"
PYTHON_BIN="${PYTHON_BIN:-/2024233240/miniconda3/envs/wamflow/bin/python}"
LIBERO_PYTHON_BIN="${LIBERO_PYTHON_BIN:-/2024233240/miniconda3/envs/libero/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-3}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
RUN_TAG="${RUN_TAG:-libero_readout_5k_seed${SEED}}"
PILOT_STEPS="${PILOT_STEPS:-5000}"
RAW_MAX_STEPS="${RAW_MAX_STEPS:-8679}"
NUM_TRIALS="${NUM_TRIALS:-10}"
BASE_CKPT="${BASE_CKPT:-/2024233240/FastWAM/runs/libero_uncond_2cam224_1e-4/libero_4xh20_bs32_wandb_20k_2026-06-12/checkpoints/weights/step_020000.pt}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/libero_readout_5k_pilots/logs}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/evaluate_results/libero_readout_5k_pilots}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-/2024233240/if-wam_incoming/ifwam_data/text_embeds_cache}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/2024233240/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"

mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"
export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
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
    echo "[readout-5k] missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[readout-5k] missing required directory: ${path}" >&2
    exit 1
  fi
}

prepare_run_dir() {
  local dir="$1"
  if [[ "${dir}" != "${ROOT}/runs/"* ]]; then
    echo "[readout-5k] refusing to clean unexpected run dir: ${dir}" >&2
    exit 1
  fi
  rm -rf "${dir}"
  mkdir -p "$(dirname "${dir}")"
}

train_stage() {
  local label="$1" cfg="$2" max_steps="$3" resume_path="$4" group="$5"
  local run_id="${RUN_TAG}_${label}"
  local run_dir
  run_dir="$(run_dir_for "${cfg}" "${run_id}")"
  local log_file="${LOG_ROOT}/${run_id}.log"
  prepare_run_dir "${run_dir}"
  require_file "${resume_path}"

  echo "[readout-5k] train start label=${label} cfg=${cfg} steps=${max_steps} resume=${resume_path}"
  RUN_ID="${run_id}" bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
    "task=${cfg}" \
    "seed=${SEED}" \
    "max_steps=${max_steps}" \
    "resume=${resume_path}" \
    "wandb.enabled=${WANDB_ENABLED}" \
    "wandb.name=${run_id}" \
    "wandb.group=${group}" \
    2>&1 | tee "${log_file}"

  local final_ckpt="${run_dir}/checkpoints/weights/$(step_tag "${max_steps}")"
  require_file "${final_ckpt}"
  ln -sfn "${final_ckpt}" "${run_dir}/checkpoints/weights/final.pt"
  echo "[readout-5k] train done label=${label} final=${final_ckpt}"
}

raw_grounding() {
  local label="$1" pilot_cfg="$2"
  local pilot_run_id="${RUN_TAG}_${label}"
  local pilot_ckpt
  pilot_ckpt="$(run_dir_for "${pilot_cfg}" "${pilot_run_id}")/checkpoints/weights/$(step_tag "${PILOT_STEPS}")"
  train_stage "${label}_raw_grounding" "ifwam_libero_gridfm_raw_grounding_1epoch" "${RAW_MAX_STEPS}" "${pilot_ckpt}" "libero-only-5k-readout-raw-grounding"
}

screening_eval() {
  local label="$1"
  local raw_run_id="${RUN_TAG}_${label}_raw_grounding"
  local ckpt
  ckpt="$(run_dir_for "ifwam_libero_gridfm_raw_grounding_1epoch" "${raw_run_id}")/checkpoints/weights/final.pt"
  local output_dir="${EVAL_ROOT}/${raw_run_id}"
  local log_file="${EVAL_ROOT}/${raw_run_id}_manager.log"
  require_file "${ckpt}"
  rm -rf "${output_dir}"
  mkdir -p "${EVAL_ROOT}"

  echo "[readout-5k] screening eval label=${label} ckpt=${ckpt} output=${output_dir}"
  PYTHON_BIN="${LIBERO_PYTHON_BIN}" "${LIBERO_PYTHON_BIN}" experiments/libero/run_libero_manager.py \
    task=libero_uncond_2cam224_1e-4 \
    ckpt="${ckpt}" \
    seed="${SEED}" \
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
  echo "[readout-5k] eval done label=${label} summary=${output_dir}/summary.csv"
}

preflight() {
  require_file "${PYTHON_BIN}"
  require_file "${LIBERO_PYTHON_BIN}"
  require_file "${BASE_CKPT}"
  require_file "${DATASET_STATS_PATH}"
  require_dir "${TEXT_CACHE_DIR}"
}

preflight

declare -A CFGS=(
  [R1]="original_readout_5k"
  [R2]="readout_direction_presence_5k"
)

for label in R1 R2; do
  train_stage "${label}" "${CFGS[$label]}" "${PILOT_STEPS}" "${BASE_CKPT}" "libero-only-5k-readout"
done

for label in R1 R2; do
  raw_grounding "${label}" "${CFGS[$label]}"
done

for label in R1 R2; do
  screening_eval "${label}"
done

echo "[readout-5k] complete RUN_TAG=${RUN_TAG}"
echo "[readout-5k] eval: ${EVAL_ROOT}"
