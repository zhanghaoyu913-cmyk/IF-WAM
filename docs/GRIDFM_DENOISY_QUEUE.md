# GridFM Denoisy Queue

This queue runs the grid-flow denoising variant as a same-setting comparison against the current IF-WAM grid-readout alignment runs.

## Objective

Compare two ways of using the same grid-flow teacher:

- Readout alignment: supervise intermediate readout heads with grid flow.
- GridFM denoising: add noisy grid-flow tokens to the MoT stack and train a grid-flow denoising loss.

The comparison keeps the current subsampled setting fixed. It is not a Fast-WAM dense-transition reproduction.

## Setting

- Seed: `0` by default.
- GPUs: `NPROC_PER_NODE=4`.
- Global batch: `batch_size 2 * gradient_accumulation_steps 4 * world_size 4 = 32`.
- Image/video: single-view current subsampled RGB, `224`, `9` frames, `action_horizon=32`.
- Stage A: grid-flow pretrain for `20000` steps.
- Stage B: raw/full LIBERO action grounding for `2000` steps from the Stage A checkpoint.
- Grounding/eval uses the FastWAM raw LIBERO LeRobot transition data with two cameras:
  `num_frames=33`, `action_video_freq_ratio=4`, final model video has 9 frames and shape `[3, 9, 224, 448]`.
- Grounding/eval uses the fixed FastWAM LIBERO 2cam stats:
  `/2024233240/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`.
- Checkpoint policy: save no periodic checkpoints and keep only the final weight file plus `final.pt` symlink.
- WandB: online by default through `WANDB_ENABLED=true`.

## Runs

The queue is `scripts/run_gridfm_denoisy_queue.sh`:

1. `ifwam_libero_gridfm_pretrain_20k`
2. `ifwam_libero_gridfm_raw_grounding_2k`
3. `ifwam_mixed_gridfm_pretrain_20k`
4. `ifwam_mixed_gridfm_raw_grounding_2k`
5. raw/full LIBERO eval for both final grounding checkpoints.

It waits by default for these tmux sessions to finish before using GPUs:

- `current_subsampled_train`
- `corrected_current_subsampled`
- `fastwam_singleview_eval_v2`
- `libero_test_v3`

Start command:

```bash
tmux new-session -d -s gridfm_denoisy_queue \
  "cd /2024233240/if-wam_gridfm && bash scripts/run_gridfm_denoisy_queue.sh 2>&1 | tee runs/gridfm_denoisy_queue/manager.log"
```

## Final Checkpoints

After the queue finishes:

- `/2024233240/if-wam_gridfm/runs/ifwam_libero_gridfm_raw_grounding_2k/gridfm_denoisy_seed0_ifwam_libero_gridfm_raw_grounding_2k/checkpoints/weights/final.pt`
- `/2024233240/if-wam_gridfm/runs/ifwam_mixed_gridfm_raw_grounding_2k/gridfm_denoisy_seed0_ifwam_mixed_gridfm_raw_grounding_2k/checkpoints/weights/final.pt`

Eval summaries are written under:

- `/2024233240/if-wam_gridfm/evaluate_results/gridfm_raw_libero/`

These checkpoints are evaluated with the same raw/full LIBERO 2cam eval path used by the corrected three-way comparison.
