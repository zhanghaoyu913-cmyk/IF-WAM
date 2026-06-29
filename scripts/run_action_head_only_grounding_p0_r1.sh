#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/2024233240/if-wam_gridfm}"
IFWAM_ROOT="${IFWAM_ROOT:-/2024233240/if-wam_incoming/if-wam}"
LIBERO_PYTHON_BIN="${LIBERO_PYTHON_BIN:-/2024233240/miniconda3/envs/libero/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-3}"
NUM_TRIALS="${NUM_TRIALS:-10}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
RAW_CFG="${RAW_CFG:-ifwam_libero_gridfm_raw_grounding_action_head_only}"
RUN_TAG="${RUN_TAG:-action_head_only_seed${SEED}}"
RAW_MAX_STEPS="${RAW_MAX_STEPS:-8679}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/action_head_only_grounding/logs}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/evaluate_results/action_head_only_grounding}"
REPORT_ROOT="${REPORT_ROOT:-${ROOT}/reports/action_head_only_grounding}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-/2024233240/if-wam_incoming/ifwam_data/text_embeds_cache}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/2024233240/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"

P0_CKPT="${P0_CKPT:-${ROOT}/runs/libero_p0_nogrid_continued_5k/libero_only_5k_seed0_P0/checkpoints/weights/step_005000.pt}"
R1_CKPT="${R1_CKPT:-${IFWAM_ROOT}/runs/original_readout_5k/ifwam_readout_5k_seed0_R1/checkpoints/weights/step_005000.pt}"

mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}" "${REPORT_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GIT_PYTHON_REFRESH=quiet
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/2024233240/if-wam/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

step_tag() { printf 'step_%06d.pt' "$1"; }
run_dir_for() { printf '%s/runs/%s/%s' "${ROOT}" "$1" "$2"; }
require_file() { [[ -f "$1" ]] || { echo "[action-head-only] missing file: $1" >&2; exit 1; }; }
run_or_print() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    eval "$@"
  fi
}

record_step0_checkpoint() {
  local run_dir="$1" ckpt="$2"
  run_or_print "mkdir -p '${run_dir}/checkpoints/weights' && ln -sfn '${ckpt}' '${run_dir}/checkpoints/weights/step_000000.pt'"
}

run_one() {
  local label="$1" ckpt="$2"
  local run_id="${RUN_TAG}_${label}"
  local run_dir
  run_dir="$(run_dir_for "${RAW_CFG}" "${run_id}")"
  require_file "${ckpt}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    rm -rf "${run_dir}"
  fi
  record_step0_checkpoint "${run_dir}" "${ckpt}"
  run_or_print "RUN_ID='${run_id}' bash scripts/train_zero1.sh '${NPROC_PER_NODE}' task='${RAW_CFG}' seed='${SEED}' max_steps='${RAW_MAX_STEPS}' resume='${ckpt}' wandb.enabled='${WANDB_ENABLED}' wandb.name='${run_id}' wandb.group='action-head-only-grounding' 2>&1 | tee '${LOG_ROOT}/${run_id}.log'"

  local final_ckpt="${run_dir}/checkpoints/weights/$(step_tag "${RAW_MAX_STEPS}")"
  local out="${EVAL_ROOT}/${run_id}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "${final_ckpt}"
  fi
  run_or_print "'${LIBERO_PYTHON_BIN}' experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 ckpt='${final_ckpt}' seed='${SEED}' MULTIRUN.num_gpus='${NUM_GPUS}' MULTIRUN.max_tasks_per_gpu='${MAX_TASKS_PER_GPU}' EVALUATION.num_trials='${NUM_TRIALS}' EVALUATION.output_dir='${out}' EVALUATION.dataset_stats_path='${DATASET_STATS_PATH}' +EVALUATION.text_embedding_cache_dir='${TEXT_CACHE_DIR}' EVALUATION.num_inference_steps=10 model.load_text_encoder=false model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null 2>&1 | tee '${EVAL_ROOT}/${run_id}_manager.log'"
  if [[ "${DRY_RUN}" != "1" ]]; then
    cp "${out}/summary.csv" "${REPORT_ROOT}/${label}_summary.csv"
  fi
}

run_one P0 "${P0_CKPT}"
run_one R1 "${R1_CKPT}"

echo "[action-head-only] complete DRY_RUN=${DRY_RUN}"
