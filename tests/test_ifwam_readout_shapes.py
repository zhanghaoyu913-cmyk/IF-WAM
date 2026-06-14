import pytest
import torch

from ifwam.models import FlowScoringHead, GridFlowReadout, ProcessFlowReadout


def test_readout_shapes():
    b, sv, a, dv, da, k = 2, 294, 32, 3072, 1024, 8
    video_mid = torch.randn(b, sv, dv)
    action_mid = torch.randn(b, a, da)
    readout = ProcessFlowReadout(
        enabled=True,
        num_flow_windows=k,
        video_dim=dv,
        action_dim=da,
        hidden_dim=128,
        text_dim=4096,
        pooling_type="role_query",
        use_grid_readout=False,
        grid_size=(8, 8),
    )
    out = readout(video_mid, action_mid)
    assert out["pred_vflow"].shape == (b, k, 3, 3)
    assert out["pred_aflow"].shape == (b, k, 3, 3)
    score = FlowScoringHead(video_dim=dv, action_dim=da, hidden_dim=128)(video_mid, action_mid, out["pred_vflow"], out["pred_aflow"])
    assert score["flow_score"].shape == (b,)


def test_grid_readout_uses_window_pairs():
    b, frames, ht, wt, dim, k = 2, 3, 4, 4, 32, 2
    video_mid = torch.randn(b, frames * ht * wt, dim)
    readout = GridFlowReadout(video_dim=dim, hidden_dim=16, num_flow_windows=k, grid_size=(8, 8), use_text=False)

    flow, motion_logit, aux = readout(
        video_mid,
        video_tokens_per_frame=ht * wt,
        video_grid_size=(frames, ht, wt),
    )

    assert flow.shape == (b, k, 8, 8, 3)
    assert motion_logit.shape == (b, k, 8, 8)
    assert aux["grid_pooling"] == "spatial_cell_window_delta"
    assert aux["window_frame_pairs"] == [(0, 1), (1, 2)]


def test_grid_readout_rejects_missing_window_endpoint():
    b, frames, ht, wt, dim, k = 1, 2, 4, 4, 32, 2
    video_mid = torch.randn(b, frames * ht * wt, dim)
    readout = GridFlowReadout(video_dim=dim, hidden_dim=16, num_flow_windows=k, grid_size=(8, 8), use_text=False)

    with pytest.raises(ValueError, match="start/end latent pair"):
        readout(video_mid, video_tokens_per_frame=ht * wt, video_grid_size=(frames, ht, wt))
