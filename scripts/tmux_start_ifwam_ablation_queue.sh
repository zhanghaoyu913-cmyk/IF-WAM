#!/usr/bin/env bash
set -euo pipefail
SESSION="${SESSION:-ifwam_ablation_queue}"
QUEUE_ID="${QUEUE_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ABLATION_STEPS="${ABLATION_STEPS:-1000}"
SAVE_WEIGHTS_EVERY="${SAVE_WEIGHTS_EVERY:-500}"
SAVE_STATE_EVERY="${SAVE_STATE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
QUEUE_GROUP="${QUEUE_GROUP:-ifwam-ablation-v1}"
LOG_DIR="${LOG_DIR:-}"
cmd="cd /2024233240/if-wam && QUEUE_ID=${QUEUE_ID} NPROC_PER_NODE=${NPROC_PER_NODE} ABLATION_STEPS=${ABLATION_STEPS} SAVE_WEIGHTS_EVERY=${SAVE_WEIGHTS_EVERY} SAVE_STATE_EVERY=${SAVE_STATE_EVERY} LOG_EVERY=${LOG_EVERY} QUEUE_GROUP=${QUEUE_GROUP}"
if [[ -n "${LOG_DIR}" ]]; then
  cmd="${cmd} LOG_DIR=${LOG_DIR}"
fi
cmd="${cmd} bash scripts/run_ifwam_ablation_queue.sh"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
tmux new -d -s "${SESSION}" "${cmd}"
echo "started tmux session=${SESSION} queue_id=${QUEUE_ID}"
echo "attach: tmux attach -t ${SESSION}"
