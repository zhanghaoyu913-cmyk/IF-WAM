#!/usr/bin/env bash
set -euo pipefail
ROOT=/2024233240/if-wam
ACTION_CKPT=/2024233240/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HF_ENDPOINT=https://hf-mirror.com
cd "$ROOT"
python scripts/hf_mirror_parallel_resume.py --repo Wan-AI/Wan2.2-TI2V-5B --output-dir "$ROOT/checkpoints/Wan-AI/Wan2.2-TI2V-5B" --workers 8 \
  diffusion_pytorch_model-00001-of-00003.safetensors \
  diffusion_pytorch_model-00002-of-00003.safetensors \
  diffusion_pytorch_model-00003-of-00003.safetensors
DIFFSYNTH_SKIP_DOWNLOAD=true python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ifwam_fastwam.yaml --output "$ACTION_CKPT" --device cuda --dtype bfloat16
