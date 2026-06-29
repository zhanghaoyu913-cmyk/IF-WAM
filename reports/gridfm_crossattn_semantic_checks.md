# GridFM Cross-attention Semantic Checks

## Scope

This report records the local code-level checks after tightening the
`aux_arch=cross_attn_decoder` implementation.

No GridFM cross-attention pilot training was started as part of this check.
The running next-stage queue remains limited to no-grid/readout/grounding
diagnostics.

## Implemented Fixes

- Added explicit timestep conditioning to `GridAuxCrossAttentionDecoder`.
  The cross-attention path now receives `grid_pre["t"]` when
  `input_mode=noisy_teacher`.
- Added a grid-token key padding mask to decoder self-attention. Invalid grid
  tokens are not visible as keys/values to valid grid tokens.
- Added all-invalid safe handling for decoder self-attention masks to avoid
  NaN when every grid token in a sample is invalid.
- Froze unused `grid_expert.blocks` when `trainable_scope=dit` and
  `aux_arch=cross_attn_decoder`.
- Added `trainable_scope=action_adapter_only`, which enables
  `action_encoder`, `time_embedding`, `time_projection`, `head`, and
  `proprio_encoder` while keeping the action backbone blocks frozen.

## Tests Run

```text
PYTHONPATH=src /2024233240/miniconda3/envs/wamflow/bin/python \
  -m pytest tests/test_gridfm_one_way_aux.py tests/test_ifwam_trainable_heads.py \
  tests/test_grid_flow_dit.py tests/test_ifwam_readout_shapes.py -q
```

Result:

```text
36 passed in 5.39s
```

## Config Checks

`gridfm_crossattn_video_text_lowgrad` resolves to:

```text
input_mode: noisy_teacher
target_mode: direction_presence
quality_application: loss_only
aux_arch: cross_attn_decoder
context_sources: video_text
shared_grad_scale: 0.1
```

`ifwam_libero_gridfm_raw_grounding_action_adapter_only` resolves with:

```text
trainable_scope: action_adapter_only
```

## Remaining Checks Before Training

The full-model production checks with the real 20k checkpoint still need to be
rerun before launching a cross-attention GridFM pilot:

- correct/shuffled/zero/absent grid primary invariance;
- `shared_grad_scale=0` blocks grid-loss gradients to primary backbone;
- `shared_grad_scale=0.1` scales primary backbone gradients relative to
  `shared_grad_scale=1.0`;
- grid head and aux decoder gradients are not scaled down.
