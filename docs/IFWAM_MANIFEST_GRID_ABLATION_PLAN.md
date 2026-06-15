# IF-WAM Manifest Grid Ablation Plan

## Scope

These experiments are IF-WAM internal manifest-window ablations. They are not
Fast-WAM same-setting runs because they do not use Fast-WAM dense LeRobot
transition-anchor sampling.

Fast-WAM remains an external baseline. Strict Fast-WAM data-setting controls are
reserved for future F1/F2 runs.

## Main Questions

1. Does mixed manifest data improve LIBERO benchmark performance?
2. Does the grid-flow teacher improve performance?
3. Is there an interaction gain between mixed data and grid-flow supervision?

## Main Experiments

- E1 `ifwam_libero_manifest_action_18k_g2k`
  - Stage A: LIBERO-only manifest, no grid, 18000 updates.
  - Stage B: LIBERO-only manifest, no grid, 2000 grounding updates.
- E2 `ifwam_libero_manifest_grid_18k_g2k`
  - Stage A: LIBERO-only manifest, grid enabled, 18000 updates.
  - Stage B: LIBERO-only manifest, no grid, 2000 grounding updates.
- E3 `ifwam_mixed_manifest_action_18k_g2k`
  - Stage A: mixed manifest, no grid, 18000 updates.
  - Stage B: LIBERO-only manifest, no grid, 2000 grounding updates.
- E4 `ifwam_mixed_manifest_grid_18k_g2k`
  - Stage A: mixed manifest, grid enabled, 18000 updates.
  - Stage B: LIBERO-only manifest, no grid, 2000 grounding updates.

All main experiments use max steps as the training control. Epoch is logged only
as a derived diagnostic.

## Fixed Controls

- Global batch size: 32.
- Image size: 224.
- Camera mode: concat.
- Video frames: 9.
- Action horizon: 32.
- Sampler: layout-homogeneous, proportional to row count, shuffled, drop-last.
- Same model initialization, optimizer family, scheduler, precision, freeze
  policy, and random seed within each seed set.
- Stage B resets optimizer and uses learning rate 0.2x Stage A.

## Reported Contrasts

- E2 - E1: grid flow on LIBERO-only manifest data.
- E3 - E1: mixed data without grid.
- E4 - E3: grid flow under mixed pretraining.
- E4 - E1: mixed data plus grid total effect.
- (E4 - E3) - (E2 - E1): interaction effect.

## Future Fairness Controls

- F1 `ifwam_libero_dense_transition_action_20k`
- F2 `ifwam_libero_dense_transition_grid_20k`

F1/F2 must align to Fast-WAM dense LeRobot transition-anchor sampling:
full LIBERO, approximately 277713 anchors, raw 33-frame window, 9 video frames
after stride 4, 32 action horizon, image+wrist horizontal concat, Fast-WAM
processor/stats/action normalizer, ordinary random sampler, global batch 32,
and 20000 max steps.
