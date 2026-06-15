#!/usr/bin/env bash
set -euo pipefail
SESSION="${SESSION:-ifwam_manifest_grid_ablation}"
ROOT="${ROOT:-/2024233240/if-wam_incoming/if-wam}"
MODE="${MODE:-debug}"
SEEDS="${SEEDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
cmd="cd ${ROOT} && MODE=${MODE} SEEDS='${SEEDS}' NPROC_PER_NODE=${NPROC_PER_NODE} bash scripts/run_ifwam_manifest_grid_ablation_queue.sh"
tmux new -d -s "${SESSION}" "${cmd}"
echo "started tmux session=${SESSION} mode=${MODE} seeds=${SEEDS}"
echo "attach: tmux attach -t ${SESSION}"
