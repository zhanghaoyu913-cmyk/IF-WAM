# Grid Flow Matching Variant

This worktree implements a separate IF-WAM variant where grid flow is trained as a flow-matching auxiliary modality instead of using the old middle-layer process-flow readout.

## Scope

- Training uses video, grid, and action tokens in the MoT stack.
- Grid flow teacher is loaded from the existing manifest `grid_flow_teacher`.
- Grid flow is noised with an independent continuous flow-matching scheduler.
- The grid expert is `GridFlowDiT`. It predicts the grid denoising target and
  contributes `lambda_gridflow_fm * loss_gridflow_fm`.
- `GridFlowDiT` uses grid-specific `grid_encoder`, `grid_head`, and
  `grid_position_embedding`. Its text/time/transformer backbone can still load
  the existing ActionDiT backbone checkpoint; only the grid-specific input,
  output, and position layers stay randomly initialized.
- Action tokens do not attend to grid tokens. Grid flow shapes the video-side
  representation through video-grid interaction, not by providing a teacher-only
  action input.
- The old `ProcessFlowReadout` and `FlowScoringHead` are disabled in the new configs.
- Inference remains the original Fast-WAM/IF-WAM action inference path: first-frame video plus action diffusion. It does not require, create, expose, or consume grid flow.
- The queue grounds and evaluates the GridFM checkpoints with the corrected raw/full LIBERO 2cam pipeline, not the older manifest-only grounding path.

## Masking

The mixed attention mask remains causal and follows the GridFM-v1 rule:
video and grid interact, while action remains video-only with respect to grid.

- Video keeps the existing first-frame-causal video mask.
- Grid window `k` can attend to video tokens up to the aligned latent prefix and grid windows up to `k`.
- Video tokens can attend only to grid windows whose transition has already ended relative to that video latent frame.
- Action tokens can attend to video prefixes aligned to their action segment.
- Action tokens cannot attend to any grid token.
- Action self-attention is unchanged from Fast-WAM because action diffusion denoises the full horizon jointly.

In block form:

```text
video  -> video + causal-past grid
grid   -> causal-prefix video + causal-past grid
action -> aligned video + action
action -/-> grid
```

## Configs

- `configs/task/ifwam_libero_gridfm_pretrain_20k.yaml`
- `configs/task/ifwam_mixed_gridfm_pretrain_20k.yaml`

Both configs use the dense subsampled manifests under:

- `/2024233240/if-wam_incoming/ifwam_data/manifests/dense_ablation/`

## Important Limitation

This variant tests whether training-time grid-flow denoising improves the shared MoT representation and downstream action prediction. It is not a test-time grid-flow-conditioned policy. A test-time grid-conditioned policy would need a separate design because benchmark observations do not provide grid flow.
