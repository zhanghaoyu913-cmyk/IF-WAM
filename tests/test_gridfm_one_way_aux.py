import warnings

import torch
import torch.nn.functional as F

from fastwam.models.wan22.fastwam import FastWAM, GridAuxCrossAttentionDecoder
from fastwam.models.wan22.grid_flow_dit import GridFlowDiT
from fastwam.models.wan22.mot import MoT


def _expert(presence=False):
    return GridFlowDiT(
        hidden_dim=16,
        flow_dim=3,
        ffn_dim=32,
        text_dim=8,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
        num_flow_windows=1,
        grid_size=(2, 2),
        predict_presence=presence,
    )


def _pre(expert, tokens):
    bsz = tokens.shape[0]
    return expert.pre_dit(
        grid_tokens=tokens,
        timestep=torch.full((bsz,), 0.5),
        context=torch.randn(bsz, 3, 8),
        context_mask=torch.ones(bsz, 3, dtype=torch.bool),
    )


def _run_mot(grid_tokens=None, legacy=False, grid_key_gate=None):
    torch.manual_seed(7)
    video = _expert()
    action = _expert()
    grid = _expert()
    mot = MoT({"video": video, "grid": grid, "action": action}, mot_checkpoint_mixed_attn=False).eval()
    for module in (video, action, grid):
        module.eval()

    bsz = 2
    video_pre = _pre(video, torch.randn(bsz, 4, 3))
    action_pre = _pre(action, torch.randn(bsz, 3, 3))
    embeds = {"video": video_pre["tokens"], "action": action_pre["tokens"]}
    freqs = {"video": video_pre["freqs"], "action": action_pre["freqs"]}
    ctx = {
        "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
        "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
    }
    tmods = {"video": video_pre["t_mod"], "action": action_pre["t_mod"]}

    if grid_tokens is None:
        mask = torch.ones(7, 7, dtype=torch.bool)
    else:
        grid_pre = _pre(grid, grid_tokens)
        embeds = {"video": video_pre["tokens"], "grid": grid_pre["tokens"], "action": action_pre["tokens"]}
        freqs["grid"] = grid_pre["freqs"]
        ctx["grid"] = {"context": grid_pre["context"], "mask": grid_pre["context_mask"]}
        tmods["grid"] = grid_pre["t_mod"]
        mask = torch.ones(11, 11, dtype=torch.bool)
        if not legacy:
            mask[:4, 4:8] = False
            mask[8:, 4:8] = False
        if grid_key_gate is not None:
            batch_mask = mask.unsqueeze(0).unsqueeze(1).expand(bsz, 1, -1, -1).clone()
            batch_mask[:, :, :, 4:8] &= grid_key_gate[:, None, None, :].bool()
            mask = batch_mask

    with torch.no_grad():
        out = mot(embeds, mask, freqs, ctx, tmods)
    return out["video"], out["action"]


def test_action_output_is_independent_of_grid_in_one_way_aux_mode():
    torch.manual_seed(11)
    grid = torch.randn(2, 4, 3)
    video_a, action_a = _run_mot(grid)
    video_b, action_b = _run_mot(grid.flip(0))
    video_c, action_c = _run_mot(torch.zeros_like(grid))
    video_d, action_d = _run_mot(None)

    for lhs, rhs in ((video_a, video_b), (action_a, action_b), (video_a, video_c), (action_a, action_c), (video_a, video_d), (action_a, action_d)):
        diff = (lhs - rhs).abs()
        assert diff.max().item() <= 1.0e-6
        assert diff.mean().item() <= 1.0e-7


def test_legacy_grid_changes_primary_output_observation_only():
    torch.manual_seed(11)
    grid = torch.randn(2, 4, 3)
    video_a, action_a = _run_mot(grid, legacy=True)
    video_b, action_b = _run_mot(grid.flip(0), legacy=True)
    # Legacy permits video queries to read grid keys, so this test records the
    # coupling without enforcing independence on the old mode.
    assert (video_a - video_b).abs().max().item() > 0.0
    assert torch.isfinite((action_a - action_b).abs()).all()


def test_missing_grid_teacher_does_not_change_action_output():
    torch.manual_seed(11)
    grid = torch.randn(2, 4, 3)
    gate = torch.zeros(2, 4)
    video_a, action_a = _run_mot(grid, grid_key_gate=gate)
    video_b, action_b = _run_mot(None)
    assert (video_a - video_b).abs().max().item() <= 1.0e-6
    assert (action_a - action_b).abs().max().item() <= 1.0e-6


def test_invalid_grid_cells_are_not_visible_to_primary_stream():
    torch.manual_seed(11)
    grid = torch.randn(2, 4, 3)
    gate = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    video_a, action_a = _run_mot(grid, grid_key_gate=gate)
    grid_b = grid.clone()
    grid_b[(gate == 0).bool()] = 1000.0
    video_b, action_b = _run_mot(grid_b, grid_key_gate=gate)
    assert (video_a - video_b).abs().max().item() <= 1.0e-6
    assert (action_a - action_b).abs().max().item() <= 1.0e-6


def _fastwam_stub(target_mode="direction_presence"):
    model = FastWAM.__new__(FastWAM)
    model.grid_flow_dim = 3
    model.grid_flow_num_windows = 1
    model.grid_flow_grid_size = (2, 2)
    model.grid_flow_target_mode = target_mode
    model.grid_min_motion_threshold = 1.0e-3
    model.grid_move_threshold_mad_scale = 3.0
    model.grid_presence_loss_weight = 1.0
    model.grid_direction_loss_weight = 1.0
    model.grid_presence_pos_weight_min = 0.25
    model.grid_presence_pos_weight_max = 8.0
    model.grid_quality_input_gate = True
    model.grid_quality_application = "token_and_loss"
    model.grid_quality_hard_threshold = 1.0e-6
    return model


def test_grid_loss_all_invalid_returns_zero():
    model = _fastwam_stub()
    pred = torch.randn(2, 1, 2, 2, 3)
    target = torch.randn_like(pred)
    valid = torch.zeros(2, 1, 2, 2)
    q = torch.ones(2, 1)
    per, logs = model._direction_presence_grid_loss_per_sample(pred, torch.zeros(2, 1, 2, 2), target, target, valid, q)
    assert per.abs().max().item() == 0.0
    assert torch.isfinite(logs["grid_moving_cell_ratio"])


def test_grid_loss_mask_normalization():
    model = _fastwam_stub("raw_vector_fm")
    pred = torch.zeros(1, 1, 2, 2, 3)
    target = torch.ones_like(pred)
    valid = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    q = torch.ones(1, 1)
    per = model._grid_flow_loss_per_sample(pred, target, valid, q)
    assert torch.allclose(per, torch.tensor([1.0]))


def test_grid_quality_zero_blocks_grid_auxiliary_residual():
    expert = _expert()
    grid = torch.randn(2, 4, 3)
    pre = expert.pre_dit(
        grid,
        torch.ones(2),
        torch.randn(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
        token_gate=torch.zeros(2, 4),
    )
    assert pre["tokens"].abs().max().item() == 0.0


def test_quality_loss_only_does_not_change_noisy_state_or_token_residual():
    model = _fastwam_stub()
    model.grid_quality_application = "loss_only"
    model.grid_quality_input_gate = False
    gate = model._grid_token_gate(
        torch.ones(1, 1, 2, 2),
        torch.zeros(1, 1),
        torch.ones(1),
        (1, 2, 2),
    )
    assert torch.allclose(gate, torch.ones_like(gate))


def test_all_zero_quality_returns_zero_auxiliary_loss():
    model = _fastwam_stub()
    clean = torch.randn(1, 1, 2, 2, 3)
    valid = torch.ones(1, 1, 2, 2)
    pred = torch.randn_like(clean)
    logits = torch.randn(1, 1, 2, 2)
    per, _ = model._direction_presence_grid_loss_per_sample(pred, logits, pred, clean, valid, torch.zeros(1, 1))
    assert per.item() == 0.0


def test_hard_mask_denominator_correct():
    model = _fastwam_stub("raw_vector_fm")
    model.grid_quality_application = "hard_mask"
    model.grid_quality_hard_threshold = 0.5
    pred = torch.zeros(1, 2, 1, 1, 3)
    target = torch.ones_like(pred)
    valid = torch.ones(1, 2, 1, 1)
    quality = torch.tensor([[1.0, 0.0]])
    per = model._grid_flow_loss_per_sample(pred, target, valid, quality)
    assert torch.allclose(per, torch.tensor([1.0]))


def test_loss_grid_raw_matches_weighted_component_sum_without_timestep_weight():
    model = _fastwam_stub()
    clean = torch.zeros(1, 1, 2, 2, 3)
    clean[:, :, 0, 0, 0] = 1.0
    valid = torch.ones(1, 1, 2, 2)
    pred = torch.zeros_like(clean)
    logits = torch.zeros(1, 1, 2, 2)
    per, logs = model._direction_presence_grid_loss_per_sample(pred, logits, pred, clean, valid, torch.ones(1, 1))
    expected = logs["unweighted_direction"] * model.grid_direction_loss_weight + logs["unweighted_presence"] * model.grid_presence_loss_weight
    assert torch.allclose(per.mean(), expected, atol=1e-6)


def test_direction_target_has_unit_norm_on_moving_cells():
    flow = torch.tensor([[[[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]]])
    direction = F.normalize(flow, dim=-1, eps=1.0e-6)
    moving = flow.norm(dim=-1) > 1.0e-3
    assert torch.allclose(direction.norm(dim=-1)[moving], torch.ones_like(direction.norm(dim=-1)[moving]), atol=1.0e-6)


def test_static_cells_do_not_receive_direction_loss():
    model = _fastwam_stub()
    clean = torch.zeros(1, 1, 2, 2, 3)
    clean[:, :, 0, 0, 0] = 1.0
    valid = torch.ones(1, 1, 2, 2)
    pred = torch.randn_like(clean) * 100
    target = torch.zeros_like(clean)
    per, _ = model._direction_presence_grid_loss_per_sample(pred, torch.zeros(1, 1, 2, 2), target, clean, valid, torch.ones(1, 1))
    assert torch.isfinite(per).all()


def test_presence_loss_receives_positive_and_negative_cells():
    model = _fastwam_stub()
    clean = torch.zeros(1, 1, 2, 2, 3)
    clean[:, :, 0, 0, 0] = 1.0
    valid = torch.ones(1, 1, 2, 2)
    pred = torch.zeros_like(clean)
    logits = torch.zeros(1, 1, 2, 2)
    per, logs = model._direction_presence_grid_loss_per_sample(pred, logits, pred, clean, valid, torch.ones(1, 1))
    assert per.item() > 0.0
    assert logs["grid_presence_pos_weight"].item() > 0.0


def test_raw_vector_mode_is_backward_compatible():
    model = _fastwam_stub("raw_vector_fm")
    pred = torch.zeros(1, 1, 2, 2, 3)
    target = torch.ones_like(pred)
    per = model._grid_flow_loss_per_sample(pred, target, torch.ones(1, 1, 2, 2), torch.ones(1, 1))
    assert torch.allclose(per, torch.tensor([1.0]))


def test_magnitude_unreliable_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("GridFM raw_vector_fm uses vector magnitude; magnitude_reliable=false", RuntimeWarning)
    assert any("magnitude_reliable=false" in str(w.message) for w in caught)


def test_learned_query_output_is_independent_of_teacher_input():
    expert = GridFlowDiT(
        hidden_dim=16,
        flow_dim=3,
        ffn_dim=32,
        text_dim=8,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
        num_flow_windows=1,
        grid_size=(2, 2),
        predict_presence=True,
        learned_query=True,
    ).eval()
    context = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    timestep = torch.zeros(2)
    a = expert.pre_dit(torch.randn(2, 4, 3), timestep, context, mask, use_learned_query=True)["tokens"]
    b = expert.pre_dit(torch.randn(2, 4, 3) * 100, timestep, context, mask, use_learned_query=True)["tokens"]
    assert torch.allclose(a, b, atol=1e-6)


def _scale_probe(scale=1.0, detach_action=False):
    torch.manual_seed(123)
    shared = torch.nn.Parameter(torch.randn(2, 4))
    action = torch.nn.Parameter(torch.randn(2, 4))
    head = torch.nn.Linear(4, 3, bias=False)
    head.weight.data.normal_()
    shared_for_grid = FastWAM._scale_grid_context_hidden(shared, scale=scale, detach=False)
    action_for_grid = FastWAM._scale_grid_context_hidden(action, scale=scale, detach=detach_action)
    out = head(shared_for_grid + action_for_grid)
    loss = out.square().mean()
    loss.backward()
    return (
        None if shared.grad is None else shared.grad.detach().clone(),
        None if action.grad is None else action.grad.detach().clone(),
        head.weight.grad.detach().clone(),
    )


def test_shared_grad_scale_zero_blocks_shared_backbone_grad():
    shared_grad, action_grad, head_grad = _scale_probe(scale=0.0, detach_action=False)
    assert shared_grad is None
    assert action_grad is None
    assert head_grad.abs().sum().item() > 0.0


def test_shared_grad_scale_scales_shared_backbone_grad_not_grid_head():
    full_shared, full_action, full_head = _scale_probe(scale=1.0, detach_action=False)
    low_shared, low_action, low_head = _scale_probe(scale=0.1, detach_action=False)
    assert torch.allclose(low_shared, full_shared * 0.1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(low_action, full_action * 0.1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(low_head, full_head, atol=1e-6, rtol=1e-5)


def test_detach_action_context_blocks_action_hidden_grad():
    shared_grad, action_grad, head_grad = _scale_probe(scale=0.1, detach_action=True)
    assert shared_grad is not None and shared_grad.abs().sum().item() > 0.0
    assert action_grad is None
    assert head_grad.abs().sum().item() > 0.0


def _cross_decoder_probe(scale=1.0, include_action=False, detach_action=False):
    torch.manual_seed(234)
    decoder = GridAuxCrossAttentionDecoder(
        grid_dim=8,
        video_dim=6,
        action_dim=5,
        num_heads=2,
        num_layers=1,
        ffn_dim=16,
        dropout=0.0,
    )
    head = torch.nn.Linear(8, 3, bias=False)
    grid = torch.randn(2, 4, 8)
    video = torch.nn.Parameter(torch.randn(2, 3, 6))
    action = torch.nn.Parameter(torch.randn(2, 2, 5))
    text = torch.randn(2, 3, 8)
    video_for_grid = FastWAM._scale_grid_context_hidden(video, scale=scale, detach=False)
    action_for_grid = None
    if include_action:
        action_for_grid = FastWAM._scale_grid_context_hidden(action, scale=scale, detach=detach_action)
    out = decoder(
        grid_tokens=grid,
        video_hidden=video_for_grid,
        text_context=text,
        text_mask=torch.ones(2, 4, 3, dtype=torch.bool),
        action_hidden=action_for_grid,
    )
    loss = head(out).square().mean()
    loss.backward()
    dec_grads = [p.grad.detach().clone() for p in decoder.parameters() if p.grad is not None]
    return (
        None if video.grad is None else video.grad.detach().clone(),
        None if action.grad is None else action.grad.detach().clone(),
        head.weight.grad.detach().clone(),
        dec_grads,
    )


def test_cross_attn_decoder_shared_grad_scale_zero_blocks_shared_grad():
    video_grad, action_grad, head_grad, dec_grads = _cross_decoder_probe(scale=0.0, include_action=False)
    assert video_grad is None
    assert action_grad is None
    assert head_grad.abs().sum().item() > 0.0
    assert sum(g.abs().sum().item() for g in dec_grads) > 0.0


def test_cross_attn_decoder_shared_grad_scale_scales_shared_not_head_or_decoder():
    full_video, _, full_head, full_dec = _cross_decoder_probe(scale=1.0, include_action=False)
    low_video, _, low_head, low_dec = _cross_decoder_probe(scale=0.1, include_action=False)
    assert torch.allclose(low_video, full_video * 0.1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(low_head, full_head, atol=1e-6, rtol=1e-5)
    assert len(low_dec) == len(full_dec)
    for low, full in zip(low_dec, full_dec):
        assert torch.allclose(low, full, atol=1e-6, rtol=1e-5)


def test_cross_attn_decoder_video_text_context_does_not_touch_action_hidden():
    video_grad, action_grad, head_grad, dec_grads = _cross_decoder_probe(scale=0.1, include_action=False)
    assert video_grad is not None and video_grad.abs().sum().item() > 0.0
    assert action_grad is None
    assert head_grad.abs().sum().item() > 0.0
    assert sum(g.abs().sum().item() for g in dec_grads) > 0.0


def test_cross_attn_decoder_detach_action_context_blocks_action_grad():
    video_grad, action_grad, head_grad, dec_grads = _cross_decoder_probe(scale=0.1, include_action=True, detach_action=True)
    assert video_grad is not None and video_grad.abs().sum().item() > 0.0
    assert action_grad is None
    assert head_grad.abs().sum().item() > 0.0
    assert sum(g.abs().sum().item() for g in dec_grads) > 0.0
