import torch
from torch import nn

from fastwam.trainer import Wan22Trainer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(2, 2)
        self.action_expert = nn.Module()
        self.action_expert.blocks = nn.Linear(2, 2)
        self.action_expert.head = nn.Linear(2, 2)
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
