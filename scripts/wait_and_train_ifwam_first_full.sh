#!/usr/bin/env bash
set -euo pipefail

ROOT=/2024233240/if-wam
ACTION_CKPT=/2024233240/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
PREP_LOG="$ROOT/runs/ifwam_actiondit_prepare.log"
TRAIN_LOG="$ROOT/runs/ifwam_first_full_train.log"

export PATH="/2024233240/miniconda3/envs/wamflow/bin:${PATH}"
export PYTHONPATH="/2024233240/if-wam/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT"

echo "[queue] waiting for complete ActionDiT checkpoint: $ACTION_CKPT"
while pgrep -f 'scripts/prepare_actiondit_complete.sh|scripts/preprocess_action_dit_backbone.py.*ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt' >/dev/null; do
  sleep 60
done

if [[ ! -s "$ACTION_CKPT" ]]; then
  echo "[fatal] ActionDiT preparation ended without checkpoint. See $PREP_LOG" >&2
  exit 1
fi

python - <<'PYVERIFY'
import torch
p='/2024233240/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'
x=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
required={'meta','backbone_state_dict','policy'}
missing=required-set(x)
if missing: raise RuntimeError(f'ActionDiT checkpoint missing keys: {sorted(missing)}')
meta=x['meta']
expected={'hidden_dim':1024,'ffn_dim':4096,'num_layers':30,'num_heads':24,'attn_head_dim':128,'text_dim':4096,'freq_dim':256}
for k,v in expected.items():
    if int(meta[k]) != v: raise RuntimeError(f'meta mismatch {k}: {meta[k]} != {v}')
if not x['backbone_state_dict']: raise RuntimeError('empty ActionDiT backbone_state_dict')
print('[verify] ActionDiT checkpoint complete:', len(x['backbone_state_dict']), 'backbone tensors')
PYVERIFY

python - <<'PYCACHE'
import hashlib,json,pathlib
m=pathlib.Path('/2024233240/ifwam_data/manifests/train_mixed.jsonl')
c=pathlib.Path('/2024233240/ifwam_data/text_embeds_cache')
missing=[]
for line in m.open():
 r=json.loads(line); td=pathlib.Path(r['traj_dir']); task=(td/'language.txt').read_text().strip()
 prompt=f"A video recorded from a robot's point of view executing the following instruction: {task}"
 h=hashlib.sha256(prompt.encode()).hexdigest(); p=c/f'{h}.t5_len128.wan22ti2v5b.pt'
 if not p.exists(): missing.append(str(p))
if missing: raise RuntimeError(f'missing {len(missing)} text caches; first={missing[0]}')
print('[verify] mixed manifest rows and text caches complete')
PYCACHE

echo "[queue] checks passed; starting first complete IF-WAM training"
exec bash scripts/train_ifwam_mixed_wandb.sh 1 \
  wandb.name=ifwam-mixed-v1-first-full \
  save_every=100 \
  log_every=10 \
  2>&1 | tee "$TRAIN_LOG"
