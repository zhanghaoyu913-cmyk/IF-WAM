#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_gridfm}"
PYTHON_BIN="${PYTHON_BIN:-/2024233240/miniconda3/envs/libero/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-${ROOT}/src}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
NUM_TRIALS="${NUM_TRIALS:-50}"
RUN_TAG="${RUN_TAG:-gridfm_denoisy_seed0}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/evaluate_results/gridfm_raw_libero}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-/2024233240/if-wam_incoming/ifwam_data/text_embeds_cache}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/2024233240/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"

cd "${ROOT}"
export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES
export GIT_PYTHON_REFRESH=quiet
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/2024233240/if-wam/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[gridfm-raw-eval] missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[gridfm-raw-eval] missing required directory: ${path}" >&2
    exit 1
  fi
}

run_dir_for() {
  local cfg="$1" run_id="$2"
  printf '%s/runs/%s/%s' "${ROOT}" "${cfg}" "${run_id}"
}

run_eval() {
  local label="$1" cfg="$2"
  local run_id="${RUN_TAG}_${label}"
  local ckpt
  ckpt="$(run_dir_for "${cfg}" "${run_id}")/checkpoints/weights/final.pt"
  local output_dir="${EVAL_ROOT}/${run_id}"
  local log_file="${EVAL_ROOT}/${run_id}_manager.log"

  require_file "${ckpt}"
  mkdir -p "${EVAL_ROOT}"
  rm -rf "${output_dir}"

  echo "[gridfm-raw-eval] start ${label} ckpt=${ckpt} output=${output_dir}"
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
  echo "[gridfm-raw-eval] done ${label} summary=${output_dir}/summary.csv"
}

require_file "${PYTHON_BIN}"
require_file "${DATASET_STATS_PATH}"
require_dir "${TEXT_CACHE_DIR}"

run_eval "ifwam_libero_gridfm_raw_grounding_2k" "ifwam_libero_gridfm_raw_grounding_2k"
run_eval "ifwam_mixed_gridfm_raw_grounding_2k" "ifwam_mixed_gridfm_raw_grounding_2k"

echo "[gridfm-raw-eval] complete RUN_TAG=${RUN_TAG}"
