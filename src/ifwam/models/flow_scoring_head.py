from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class FlowScoringHead(nn.Module):
    def __init__(
        self,
        video_dim: int = 3072,
        action_dim: int = 1024,
        hidden_dim: int = 1024,
        score_granularity: str = "trajectory",
        text_dim: int = 4096,
        use_text: bool = True,
        **_: Any,
    ):
        super().__init__()
        if score_granularity not in {"trajectory", "window"}:
            raise ValueError("score_granularity must be trajectory or window")
        self.score_granularity = score_granularity
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.flow_proj = nn.Linear(18, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim) if use_text else None
        self.score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.conf = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        video_mid: torch.Tensor,
        action_mid: torch.Tensor,
        pred_vflow: Optional[torch.Tensor] = None,
        pred_aflow: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        denoise_residual: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        v = self.video_proj(video_mid.mean(dim=1))
        a = self.action_proj(action_mid.mean(dim=1))
        feat = v + a
        if pred_vflow is not None and pred_aflow is not None:
            flow = torch.cat([pred_vflow.reshape(pred_vflow.shape[0], pred_vflow.shape[1], -1), pred_aflow.reshape(pred_aflow.shape[0], pred_aflow.shape[1], -1)], dim=-1)
            if self.score_granularity == "window":
                feat_w = feat.unsqueeze(1) + self.flow_proj(flow)
                return {"flow_score": self.score(feat_w).squeeze(-1), "flow_confidence": self.conf(feat_w).squeeze(-1)}
            feat = feat + self.flow_proj(flow.mean(dim=1))
        if text_context is not None and self.text_proj is not None:
            feat = feat + self.text_proj(text_context.mean(dim=1))
        return {"flow_score": self.score(feat).squeeze(-1), "flow_confidence": self.conf(feat).squeeze(-1)}
