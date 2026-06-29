import torch
from torch import nn

from fastwam.trainer import Wan22Trainer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(2, 2)
        self.action_expert = nn.Module()
        self.action_expert.action_encoder = nn.Linear(2, 2)
        self.action_expert.time_embedding = nn.Linear(2, 2)
        self.action_expert.time_projection = nn.Linear(2, 2)
        self.action_expert.blocks = nn.Linear(2, 2)
        self.action_expert.head = nn.Linear(2, 2)
        self.grid_expert = nn.Module()
        self.grid_expert.blocks = nn.Linear(2, 2)
        self.grid_expert.grid_encoder = nn.Linear(2, 2)
        self.grid_expert.time_embedding = nn.Linear(2, 2)
        self.grid_expert.time_projection = nn.Linear(2, 2)
        self.grid_expert.grid_head = nn.Linear(2, 2)
        self.proprio_encoder = nn.Linear(2, 2)
        self.process_flow_readout = nn.Linear(2, 2)
        self.flow_scoring_head = nn.Linear(2, 1)
        self.grid_aux_decoder = nn.Linear(2, 2)
        self.frozen = nn.Linear(2, 2)


def test_ifwam_heads_remain_trainable():
    model = DummyModel()
    Wan22Trainer._apply_dit_only_train_mode(model)
    params = Wan22Trainer._collect_trainable_params(model)
    ids = {id(p) for p in params}
    assert all(id(p) in ids and p.requires_grad for p in model.process_flow_readout.parameters())
    assert all(id(p) in ids and p.requires_grad for p in model.flow_scoring_head.parameters())
    assert all(id(p) in ids and p.requires_grad for p in model.grid_aux_decoder.parameters())
    assert all(not p.requires_grad for p in model.frozen.parameters())


def test_action_head_only_trainable_scope_freezes_backbone():
    model = DummyModel()
    Wan22Trainer._apply_dit_only_train_mode(model, trainable_scope="action_head_only")
    params = Wan22Trainer._collect_trainable_params(model)
    ids = {id(p) for p in params}
    assert all(id(p) in ids and p.requires_grad for p in model.action_expert.head.parameters())
    assert all(not p.requires_grad for p in model.action_expert.blocks.parameters())
    assert all(not p.requires_grad for p in model.dit.parameters())
    assert all(not p.requires_grad for p in model.process_flow_readout.parameters())


def test_cross_attn_scope_freezes_unused_grid_blocks():
    model = DummyModel()
    model.grid_aux_arch = "cross_attn_decoder"
    Wan22Trainer._apply_dit_only_train_mode(model, trainable_scope="dit")
    params = Wan22Trainer._collect_trainable_params(model)
    ids = {id(p) for p in params}
    assert all(not p.requires_grad for p in model.grid_expert.blocks.parameters())
    assert all(id(p) in ids and p.requires_grad for p in model.grid_expert.grid_encoder.parameters())
    assert all(id(p) in ids and p.requires_grad for p in model.grid_expert.time_embedding.parameters())
    assert all(id(p) in ids and p.requires_grad for p in model.grid_expert.grid_head.parameters())
    assert all(id(p) in ids and p.requires_grad for p in model.grid_aux_decoder.parameters())


def test_action_adapter_only_scope_trains_action_adapters():
    model = DummyModel()
    Wan22Trainer._apply_dit_only_train_mode(model, trainable_scope="action_adapter_only")
    params = Wan22Trainer._collect_trainable_params(model)
    ids = {id(p) for p in params}
    for module in (
        model.action_expert.action_encoder,
        model.action_expert.time_embedding,
        model.action_expert.time_projection,
        model.action_expert.head,
        model.proprio_encoder,
    ):
        assert all(id(p) in ids and p.requires_grad for p in module.parameters())
    assert all(not p.requires_grad for p in model.action_expert.blocks.parameters())
    assert all(not p.requires_grad for p in model.dit.parameters())
