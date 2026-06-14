#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:-1}"
shift || true

export WANDB_API_KEY="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"
export WANDB_MODE="online"
export PATH="/2024233240/miniconda3/envs/wamflow/bin:/root/.codex/tmp/arg0/codex-arg0Ivw6P8:/usr/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex-path:/2024233240/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
export PYTHONPATH="/2024233240/if-wam/src"

cd /2024233240/if-wam

exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=ifwam_mixed_7d_224 \
  wandb.enabled=true \
  wandb.workspace=null \
  wandb.project=if-wam \
  wandb.group=ifwam-mixed-v1 \
  wandb.mode=online \
  "$@"
