import torch

from fastwam.models.wan22.grid_flow_dit import GridFlowDiT


def test_grid_flow_dit_token_shapes():
    model = GridFlowDiT(
        hidden_dim=32,
        flow_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
        num_flow_windows=2,
        grid_size=(2, 2),
    )
    grid = torch.randn(2, 8, 3)
    context = torch.randn(2, 5, 16)
    pre = model.pre_dit(grid_tokens=grid, timestep=torch.rand(2), context=context)

    assert pre["tokens"].shape == (2, 8, 32)
    assert model.post_dit(pre["tokens"], pre).shape == (2, 8, 3)
    assert model.grid_position_embedding.shape == (1, 8, 32)


def test_grid_flow_dit_action_backbone_key_compatibility():
    with torch.device("meta"):
        model = GridFlowDiT(
            hidden_dim=32,
            flow_dim=3,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            eps=1.0e-6,
            num_heads=2,
            attn_head_dim=8,
            num_layers=1,
            num_flow_windows=2,
            grid_size=(2, 2),
        )
    skipped = set(model.state_dict()) - GridFlowDiT.backbone_key_set(model.state_dict().keys())

    assert "grid_encoder.weight" in skipped
    assert "grid_head.weight" in skipped
    assert "grid_position_embedding" in skipped
