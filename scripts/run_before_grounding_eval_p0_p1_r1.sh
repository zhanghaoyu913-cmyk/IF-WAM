#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_gridfm}"
IFWAM_ROOT="${IFWAM_ROOT:-/2024233240/if-wam_incoming/if-wam}"
LIBERO_PYTHON_BIN="${LIBERO_PYTHON_BIN:-/2024233240/miniconda3/envs/libero/bin/python}"
SEED="${SEED:-0}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-10}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-/2024233240/if-wam_incoming/ifwam_data/text_embeds_cache}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/2024233240/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/evaluate_results/libero_only_5k_pilots_before_grounding}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/libero_only_5k_pilots_before_grounding/logs}"

mkdir -p "${EVAL_ROOT}" "${LOG_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GIT_PYTHON_REFRESH=quiet
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/2024233240/if-wam/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[before-grounding-eval] missing required file: $1" >&2
    exit 1
  fi
}

eval_ckpt() {
  local label="$1"
  local ckpt="$2"
  local output_dir="${EVAL_ROOT}/${label}_before_grounding"
  local log_file="${LOG_ROOT}/${label}_before_grounding_manager.log"
  require_file "${ckpt}"
  rm -rf "${output_dir}"
  mkdir -p "${EVAL_ROOT}"
  echo "[before-grounding-eval] start label=${label} ckpt=${ckpt}"
  "${LIBERO_PYTHON_BIN}" experiments/libero/run_libero_manager.py \
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
  cp "${output_dir}/summary.csv" "${output_dir}/before_grounding_summary.csv"
  echo "[before-grounding-eval] done label=${label} summary=${output_dir}/before_grounding_summary.csv"
}

P0_CKPT="${P0_CKPT:-${ROOT}/runs/libero_p0_nogrid_continued_5k/libero_only_5k_seed0_P0/checkpoints/weights/step_005000.pt}"
P1_CKPT="${P1_CKPT:-${ROOT}/runs/gridfm_one_way_direction_presence_loss_only/libero_only_5k_seed0_P1/checkpoints/weights/step_005000.pt}"
R1_CKPT="${R1_CKPT:-${IFWAM_ROOT}/runs/original_readout_5k/ifwam_readout_5k_seed0_R1/checkpoints/weights/step_005000.pt}"

eval_ckpt "P0" "${P0_CKPT}"
eval_ckpt "P1" "${P1_CKPT}"
eval_ckpt "R1" "${R1_CKPT}"

echo "[before-grounding-eval] complete output=${EVAL_ROOT}"
