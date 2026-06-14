# IF-WAM Integration Notes

This worktree is for adding object-centric 3D supervision to FastWAM while leaving the original `../FastWAM` checkout untouched.

## Scope

Keep `AutoLabel-3D_Affordance_Flow/run_pipeline` as an external producer. Do not copy TraceForge, SAM2, GroundingDINO, or auto-generated flow code into this repo. IF-WAM should consume exported teacher assets only.

This sparse worktree intentionally contains code, configs, scripts, metadata, and docs. It does not contain a local `data` directory. Dataset configs point to `../sim_data/fastwam/data/...`, while checkpoint configs still point to `../FastWAM/checkpoints/...` so the existing assets remain usable without duplication.

## 3D Teacher Assets

The 3D object-centric pipeline exports per-episode teacher assets with this stable file:

```text
teacher_targets.npz
  trace_3d:           [Nq, Ttrace, 3] float32
  trace_2d:           [Nq, Ttrace, 2] float32
  valid_steps:        [Nq, Ttrace] bool
  visibility:         [Nq, Ttrace] bool
  query_frame_index:  [Nq] int32
  role_id:            [Nq] int32
```

Optional sidecars may also exist:

```text
hand_flow.npy
tool_flow.npy
target_flow.npy
query_role_id.npy
query_role_name.json
query_frame_index.npy
teacher_quality.json
```

Only `teacher_targets.npz` should be required by FastWAM training. Sidecars are useful for debugging and quality gating.

## FastWAM Insertion Points

The token surfaces to supervise are explicit:

- Video tokens are created in `WanVideoDiT.pre_dit` with shape `[B, Sv, 3072]`.
- Action tokens are created in `ActionDiT.pre_dit` with shape `[B, A, 1024]`.
- Joint interaction happens in `MoT.forward`.
- Losses are assembled in `FastWAM.training_loss` and variant overrides such as `FastWAMIDM.training_loss`.

Recommended implementation order:

1. Extend `RobotVideoDataset` to optionally return `teacher_targets`.
2. Add config fields for `teacher_target_root`, quality filtering, role selection, and loss weights.
3. Add lightweight projection heads on top of selected video/action hidden states.
4. Compute masked losses with `visibility` and `valid_steps`.
5. Log separate `loss_if_video`, `loss_if_action`, and `loss_if_total` entries.

## Temporal Alignment

Current FastWAM defaults are:

```text
raw frames:       33
video frames:      9
video latent Tz:   3
action horizon:   32
```

Teacher traces may be denser than video tokens. The first implementation should align by frame index:

- use `query_frame_index` for query start frame,
- resample or bucket `trace_*` onto the 9 sampled video frames for video-token supervision,
- align action-token supervision over the 32 action steps.

Avoid assuming TraceForge output time length equals either 9 or 32.

## Files Expected To Change

Primary:

- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/fastwam_joint.py`
- `src/fastwam/models/wan22/fastwam_idm.py`
- `src/fastwam/runtime.py`
- `configs/model/*.yaml`
- `configs/data/*.yaml`

Likely secondary:

- `src/fastwam/models/wan22/mot.py` if layer-level hidden states are needed.
- `src/fastwam/models/wan22/action_dit.py` if action-token heads should live inside the expert.
- `src/fastwam/models/wan22/wan_video_dit.py` if video-token heads should live inside the expert.
