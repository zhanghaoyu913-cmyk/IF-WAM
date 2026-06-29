#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_gridfm}"
PYTHON_BIN="${PYTHON_BIN:-/2024233240/miniconda3/envs/wamflow/bin/python}"
LIBERO_PYTHON_BIN="${LIBERO_PYTHON_BIN:-/2024233240/miniconda3/envs/libero/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-10}"
SEED="${SEED:-0}"
M0_SEED="${M0_SEED:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
DRY_RUN="${DRY_RUN:-1}"
RUN_TAG="${RUN_TAG:-libero_next_stage}"
INCLUDE_CROSSATTN="${INCLUDE_CROSSATTN:-0}"
PILOT_STEPS="${PILOT_STEPS:-5000}"
RAW_MAX_STEPS="${RAW_MAX_STEPS:-8679}"
BASE_CKPT="${BASE_CKPT:-/2024233240/FastWAM/runs/libero_uncond_2cam224_1e-4/libero_4xh20_bs32_wandb_20k_2026-06-12/checkpoints/weights/step_020000.pt}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/libero_next_stage_5k_pilots/logs}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/evaluate_results/libero_next_stage_5k_pilots}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-/2024233240/if-wam_incoming/ifwam_data/text_embeds_cache}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/2024233240/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"

mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GIT_PYTHON_REFRESH=quiet
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/2024233240/if-wam/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

step_tag() { printf 'step_%06d.pt' "$1"; }
run_dir_for() { printf '%s/runs/%s/%s' "${ROOT}" "$1" "$2"; }
require_file() { [[ -f "$1" ]] || { echo "[next-stage] missing required file: $1" >&2; exit 1; }; }
run_or_print() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    eval "$@"
  fi
}

train_stage() {
  local label="$1" cfg="$2" seed="$3" resume_path="$4" steps="${5:-${PILOT_STEPS}}" run_suffix="${6:-${label}}"
  local run_id="${RUN_TAG}_seed${seed}_${run_suffix}"
  local run_dir
  run_dir="$(run_dir_for "${cfg}" "${run_id}")"
  if [[ "${DRY_RUN}" != "1" || -f "${resume_path}" ]]; then
    require_file "${resume_path}"
  else
    echo "[next-stage] DRY_RUN allows future resume path: ${resume_path}"
  fi
  if [[ "${DRY_RUN}" != "1" ]]; then
    rm -rf "${run_dir}"
  fi
  run_or_print "RUN_ID='${run_id}' bash scripts/train_zero1.sh '${NPROC_PER_NODE}' task='${cfg}' seed='${seed}' max_steps='${steps}' resume='${resume_path}' wandb.enabled='${WANDB_ENABLED}' wandb.name='${run_id}' wandb.group='libero-next-stage-5k' 2>&1 | tee '${LOG_ROOT}/${run_id}.log'"
  if [[ "${DRY_RUN}" != "1" ]]; then
    local ckpt="${run_dir}/checkpoints/weights/$(step_tag "${steps}")"
    require_file "${ckpt}"
    ln -sfn "${ckpt}" "${run_dir}/checkpoints/weights/final.pt"
  fi
}

raw_grounding() {
  local label="$1" cfg="$2" seed="$3"
  local pilot_run_id="${RUN_TAG}_seed${seed}_${label}"
  local pilot_ckpt="$(run_dir_for "${cfg}" "${pilot_run_id}")/checkpoints/weights/$(step_tag "${PILOT_STEPS}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[next-stage] raw grounding would use ${pilot_ckpt}"
  fi
  train_stage "${label}" "ifwam_libero_gridfm_raw_grounding_1epoch" "${seed}" "${pilot_ckpt}" "${RAW_MAX_STEPS}" "${label}_raw_grounding"
}

screening_eval() {
  local label="$1" seed="$2"
  local raw_run_id="${RUN_TAG}_seed${seed}_${label}_raw_grounding"
  local ckpt="$(run_dir_for "ifwam_libero_gridfm_raw_grounding_1epoch" "${raw_run_id}")/checkpoints/weights/final.pt"
  local output_dir="${EVAL_ROOT}/${raw_run_id}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "${ckpt}"
  fi
  if [[ "${DRY_RUN}" != "1" ]]; then
    rm -rf "${output_dir}"
  fi
  run_or_print "'${LIBERO_PYTHON_BIN}' experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 ckpt='${ckpt}' seed='${SEED}' MULTIRUN.num_gpus='${NUM_GPUS}' MULTIRUN.max_tasks_per_gpu='${MAX_TASKS_PER_GPU}' EVALUATION.num_trials='${NUM_TRIALS}' EVALUATION.output_dir='${output_dir}' EVALUATION.dataset_stats_path='${DATASET_STATS_PATH}' +EVALUATION.text_embedding_cache_dir='${TEXT_CACHE_DIR}' EVALUATION.num_inference_steps=10 model.load_text_encoder=false model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null 2>&1 | tee '${EVAL_ROOT}/${raw_run_id}_manager.log'"
}

declare -A CFGS=(
  [M0]="libero_p0_nogrid_continued_5k_seed1"
  [M2]="readout_original_lowlambda"
)
declare -A SEEDS=(
  [M0]="${M0_SEED}"
  [M2]="${SEED}"
)
if [[ "${INCLUDE_CROSSATTN}" == "1" ]]; then
  CFGS[M1]="gridfm_crossattn_video_text_lowgrad"
  SEEDS[M1]="${SEED}"
fi

require_file "${BASE_CKPT}"
labels=(M0 M2)
if [[ "${INCLUDE_CROSSATTN}" == "1" ]]; then
  labels+=(M1)
fi
for label in "${labels[@]}"; do
  train_stage "${label}" "${CFGS[$label]}" "${SEEDS[$label]}" "${BASE_CKPT}"
done
for label in "${labels[@]}"; do
  raw_grounding "${label}" "${CFGS[$label]}" "${SEEDS[$label]}"
done
for label in "${labels[@]}"; do
  screening_eval "${label}" "${SEEDS[$label]}"
done

echo "[next-stage] complete DRY_RUN=${DRY_RUN}"
