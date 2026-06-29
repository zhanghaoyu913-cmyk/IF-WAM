#!/usr/bin/env bash
set -euo pipefail

PIPELINE_ROOT="${PIPELINE_ROOT:-/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow/run_pipeline/ifwam}"
PYTHON_BIN="${PYTHON_BIN:-/2024233240/miniconda3/envs/wamflow/bin/python}"
TRACEFORGE_ROOT="${TRACEFORGE_ROOT:-/2024233240/if-wam_incoming/traceforge_dense_outputs}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/2024233240/if-wam_incoming/ifwam_data/manifests/dense_ablation/train_libero_dense_f4_s1_grid_rgb_end_local_short.jsonl}"
LOG_ROOT="${LOG_ROOT:-/2024233240/if-wam_gridfm/runs/libero_end_local_short_teacher_generation/logs}"

mkdir -p "${LOG_ROOT}"
cd "${PIPELINE_ROOT}"

DATASET_SPECS=(
  "/2024233240/if-wam_incoming/sim_data/libero_spatial_rvideo_dataset|${TRACEFORGE_ROOT}/libero_spatial|/2024233240/sim_data/libero_spatial_rvideo_dataset=/2024233240/if-wam_incoming/sim_data/libero_spatial_rvideo_dataset"
  "/2024233240/if-wam_incoming/sim_data/libero_object_rvideo_dataset|${TRACEFORGE_ROOT}/libero_object|/2024233240/sim_data/libero_object_rvideo_dataset=/2024233240/if-wam_incoming/sim_data/libero_object_rvideo_dataset"
  "/2024233240/if-wam_incoming/sim_data/libero_goal_rvideo_dataset|${TRACEFORGE_ROOT}/libero_goal|/2024233240/sim_data/libero_goal_rvideo_dataset=/2024233240/if-wam_incoming/sim_data/libero_goal_rvideo_dataset"
  "/2024233240/sim_data/libero_10_rvideo_dataset|${TRACEFORGE_ROOT}/libero10|"
)

for spec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r root trace_root rewrite_rule <<< "${spec}"
  name="$(basename "${root}")"
  echo "[short-teacher] start ${name} trace=${trace_root}"
  rewrite_args=()
  if [[ -n "${rewrite_rule}" ]]; then
    rewrite_args=(--rewrite-traj-prefix "${rewrite_rule}")
  fi
  "${PYTHON_BIN}" generate_ifwam_flow_teachers.py \
    --dataset-root "${root}" \
    --traceforge-output-root "${trace_root}" \
    --generate-role-flow false \
    --generate-grid-flow true \
    --copy-role-to-canonical false \
    --num-flow-windows 2 \
    --grid-size 8 8 \
    --flow-frame-span 4 \
    --grid-pair-manifest "${PAIR_MANIFEST}" \
    --grid-window-mode end_local_short \
    --grid-output-name grid_flow_end_local_short \
    --overwrite false \
    "${rewrite_args[@]}" \
    2>&1 | tee "${LOG_ROOT}/${name}.log"
  echo "[short-teacher] done ${name}"
done

echo "[short-teacher] complete"
