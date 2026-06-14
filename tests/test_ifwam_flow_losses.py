import torch

from ifwam.losses.process_flow_losses import action_flow_loss, consistency_loss, direction_loss, grid_flow_loss, magnitude_loss


def test_direction_loss_finite_and_invalid_zero():
    pred = torch.randn(2, 8, 3, 3, requires_grad=True)
    target = torch.randn(2, 8, 3, 3)
    valid = torch.ones(2, 8, 3)
    loss = direction_loss(pred, target, valid)
    assert torch.isfinite(loss)
    zero = direction_loss(pred, target, torch.zeros_like(valid))
    assert torch.isfinite(zero)
    assert zero.item() == 0.0


def test_magnitude_and_consistency():
    pred = torch.zeros(2, 8, 3, 3, requires_grad=True)
    pred.data[:, :, 0] = torch.tensor([1.0, 0.0, 0.0])
    pred.data[:, :, 1] = torch.tensor([2.0, 0.0, 0.0])
    pred.data[:, :, 2] = torch.tensor([1.0, 0.0, 0.0])
    target = pred.detach().clone()
    valid = torch.ones(2, 8, 3)
    quality = torch.ones(2, 8)
    assert magnitude_loss(pred, target, valid, quality).item() < 1e-6
    assert consistency_loss(pred, valid, quality).item() < 1e-6


def test_action_flow_sigma_gating_reduces_loss():
    pred = torch.randn(2, 8, 3, 3, requires_grad=True)
    target = torch.randn(2, 8, 3, 3)
    valid = torch.ones(2, 8, 3)
    quality = torch.ones(2, 8)
    mask = {"aflow": torch.ones(2)}
    low, _ = action_flow_loss(pred, target, valid, quality, mask, torch.zeros(2), {"sigma_gamma": 2.0})
    high, _ = action_flow_loss(pred, target, valid, quality, mask, torch.ones(2), {"sigma_gamma": 2.0})
    assert high.item() <= low.item()


def test_grid_flow_loss_finite_and_masked_zero():
    pred = torch.randn(2, 8, 8, 8, 3, requires_grad=True)
    logits = torch.randn(2, 8, 8, 8, requires_grad=True)
    target = torch.randn(2, 8, 8, 8, 3)
    valid = torch.ones(2, 8, 8, 8)
    quality = torch.ones(2, 8)
    mask = {"gridflow": torch.ones(2), "mag": torch.zeros(2)}
    loss, logs = grid_flow_loss(pred, logits, target, valid, quality, mask, {})
    assert torch.isfinite(loss)
    assert "grid_motion_presence" in logs
    zero, _ = grid_flow_loss(pred, logits, target, valid, quality, {"gridflow": torch.zeros(2), "mag": torch.zeros(2)}, {})
    assert torch.isfinite(zero)
    assert zero.item() == 0.0


def test_grid_flow_adaptive_threshold_rejects_background_jitter():
    pred = torch.randn(1, 2, 8, 8, 3, requires_grad=True)
    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    target = torch.full((1, 2, 8, 8, 3), 5.0e-4)
    target[:, :, 2:4, 2:4, 0] = 2.0e-2
    valid = torch.ones(1, 2, 8, 8)
    quality = torch.ones(1, 2)
    mask = {"gridflow": torch.ones(1), "mag": torch.zeros(1)}
    cfg = {
        "move_threshold_mode": "adaptive_mad",
        "move_threshold_min": 1.0e-3,
        "move_threshold_mad_scale": 3.0,
    }
    loss, logs = grid_flow_loss(pred, logits, target, valid, quality, mask, cfg)
    assert torch.isfinite(loss)
    assert 0.0 < logs["grid_moving_cell_ratio"].item() < 0.2
    assert logs["grid_move_threshold"].item() >= 1.0e-3
