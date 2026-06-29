# GridFM Information Flow Audit

## Scope

This audit covers the GridFM training-time teacher path in `/2024233240/if-wam_gridfm` and the teacher construction path in `/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow`.

## Training Call Graph

1. The manifest builder writes training windows with latent-aligned explicit flow pairs in [build_ifwam_manifest.py](/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow/run_pipeline/ifwam/build_ifwam_manifest.py:97). `frame_indices` are subsampled by `vae_temporal_factor`; `flow_start_indices` and `flow_end_indices` are adjacent latent anchors at [build_ifwam_manifest.py](/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow/run_pipeline/ifwam/build_ifwam_manifest.py:98).
2. The teacher builder loads TraceForge `coords` and computes `grid_flow_vectors = p_end - p_start` at [grid_flow_builder.py](/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow/run_pipeline/ifwam/flow/grid_flow_builder.py:119) and [grid_flow_builder.py](/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow/run_pipeline/ifwam/flow/grid_flow_builder.py:134). `p_start` is projected into the start camera for grid assignment at [grid_flow_builder.py](/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow/run_pipeline/ifwam/flow/grid_flow_builder.py:125).
3. The dataset loader reads `grid_flow.npz`, checks `grid_assignment=reference_frame_projection`, maps requested `(start,end)` pairs, and returns `grid_flow_teacher`, `grid_flow_valid`, `grid_flow_quality`, `grid_flow_mask`, and `loss_mask` at [manifest_dataset.py](/2024233240/if-wam_gridfm/src/ifwam/data/manifest_dataset.py:265).
4. The collator stacks grid tensors and converts `loss_mask["gridflow"]` to a batch tensor at [collate.py](/2024233240/if-wam_gridfm/src/ifwam/data/collate.py:22).
5. `FastWAM.training_loss` samples video/action noise and timesteps, then samples grid noise/timestep when GridFM is enabled at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:801).
6. `raw_vector_fm` uses raw 3D vector endpoints. `direction_presence` first converts the clean endpoint to unit direction on moving valid cells at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:831).
7. `GridFlowDiT.pre_dit` maps noisy grid vectors to grid tokens, with optional token gate applied after position embedding at [grid_flow_dit.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/grid_flow_dit.py:279).
8. MoT concatenates active expert tokens, builds Q/K/V per expert, runs shared scaled-dot-product attention, then splits output back to each expert at [mot.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/mot.py:447).
9. Grid output is projected by `grid_head`, and in `direction_presence` also by `presence_head`, then loss is applied with valid/quality/sample weights at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:1005).

## Eval Call Graph

Eval/inference calls `_predict_joint_noise`, `_predict_action_noise`, or `_predict_action_noise_with_cache`. These paths build only video/action pre-DiT states and call MoT with `{"video", "action"}` only; no grid teacher, grid timestep, noisy grid, or grid token is created in inference. See [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:1128) and [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:1180).

## Query-Key Visibility Matrix

Legacy training with grid tokens:

| Query \ Key | video | action | grid |
|---|---|---|---|
| video | causal video mask | no | partially allowed |
| action | aligned video prefix | full action | no |
| grid | aligned video prefix | no | causal grid window |

The legacy video-to-grid allowance is created by [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:691). Therefore grid teacher can affect video output during training. Action does not directly attend to grid, but can be affected indirectly through video tokens in subsequent layers.

`one_way_aux` training with grid tokens:

| Query \ Key | video | action | grid |
|---|---|---|---|
| video | unchanged from primary mask | unchanged from primary mask | forbidden |
| action | unchanged from primary mask | unchanged from primary mask | forbidden |
| grid | allowed | allowed | allowed only for valid gated grid keys |

The one-way override is applied at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:896). Per-sample invalid grid keys are removed by a `[B,1,S,S]` mask at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:244). MoT accepts this 4D mask in [mot.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/mot.py:470).

## Non-Attention Coupling

- No MoE/MoT expert capacity competition was found. MoT concatenates all tokens and applies dense attention; there is no router capacity or token dropping in [mot.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/mot.py:525).
- Grid timestep is local to `GridFlowDiT.pre_dit` and does not enter video/action time modulation. Video/action timesteps are generated separately in [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:771) and [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:783).
- Grid tokens are inserted between video and action in legacy training, but action token slicing is by expert name after MoT output, not by fixed absolute index, at [mot.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/mot.py:539).
- Quality and sample masks previously affected only loss. In `one_way_aux`, they also gate grid token residual/input and key visibility through [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:226).

## Teacher Missing Path

Dataset missing path sets `loss_mask["gridflow"]=0` and returns zero teacher/valid/quality at [manifest_dataset.py](/2024233240/if-wam_gridfm/src/ifwam/data/manifest_dataset.py:276). In `one_way_aux`, a missing runtime `grid_flow_teacher` forces `grid_mask=0` at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:803), and no grid pre-DiT is created if no sample in the batch has a valid grid mask at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:875). Legacy preserves the old `ValueError` for missing teacher.

## Invalid Cell Path

Dataset invalid cells are encoded in `grid_flow_valid` at [manifest_dataset.py](/2024233240/if-wam_gridfm/src/ifwam/data/manifest_dataset.py:316). In `one_way_aux`, `grid_flow_valid`, `grid_quality`, and sample-level grid mask create `grid_token_gate` at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:226). The gate zeros token residual/input in [grid_flow_dit.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/grid_flow_dit.py:281) and removes invalid grid keys/values from batch-specific attention masks at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:244).

## Train/Eval Mismatch

Legacy has a train/eval mismatch: train has grid tokens visible to video, eval has no grid stream. `one_way_aux` removes teacher-to-primary visibility, so primary video/action outputs are invariant to grid value, grid timestep, grid quality, and grid stream existence under fixed primary inputs. This is covered by `test_action_output_is_independent_of_grid_in_one_way_aux_mode`.

## Target Modes

- `raw_vector_fm`: legacy behavior, MSE on raw vector FM target, still logs a one-time warning because teacher metadata records `magnitude_reliable=false`.
- `direction_presence`: clean endpoint is unit direction for moving valid cells only; static cells do not contribute direction loss; presence is learned by a separate head instantiated only in this mode. Moving threshold uses adaptive median/MAD over valid cells at [fastwam.py](/2024233240/if-wam_gridfm/src/fastwam/models/wan22/fastwam.py:261).

## Teacher Coordinate Frame

The builder computes displacement from TraceForge `coords`, now explicitly recorded as `flow_coordinate_frame=traceforge_world` and `flow_vector_definition=p_end_world_minus_p_start_world` at [grid_flow_builder.py](/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow/run_pipeline/ifwam/flow/grid_flow_builder.py:170). Existing teacher files generated before this change lack this field and should be flagged high-risk by `tools/audit_gridflow_teacher.py`.
