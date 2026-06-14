from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _text_summary(text_context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if text_context is None:
        return None
    if text_context.ndim != 3:
        raise ValueError(f"text_context must be [B,L,D], got {tuple(text_context.shape)}")
    return text_context.mean(dim=1)


def _prefix_end(window_idx: int, seq_len: int, video_tokens_per_frame: Optional[int]) -> int:
    if video_tokens_per_frame is None or video_tokens_per_frame <= 0:
        return seq_len
    return min(seq_len, int(window_idx + 1) * int(video_tokens_per_frame))


def _resolve_video_grid(
    seq_len: int,
    video_tokens_per_frame: Optional[int],
    video_grid_size: Optional[tuple[int, int, int]] = None,
) -> tuple[int, int, int]:
    if video_grid_size is not None:
        f, h, w = (int(video_grid_size[0]), int(video_grid_size[1]), int(video_grid_size[2]))
        if f > 0 and h > 0 and w > 0 and f * h * w == seq_len:
            return f, h, w
    if video_tokens_per_frame is None or int(video_tokens_per_frame) <= 0:
        side = int(seq_len ** 0.5)
        while side > 1 and seq_len % side != 0:
            side -= 1
        return 1, side, max(1, seq_len // max(1, side))
    tpf = int(video_tokens_per_frame)
    frames = max(1, seq_len // tpf)
    side = int(tpf ** 0.5)
    while side > 1 and tpf % side != 0:
        side -= 1
    return frames, side, max(1, tpf // max(1, side))


def _pool_spatial_grid(frame_tokens: torch.Tensor, out_grid: tuple[int, int]) -> torch.Tensor:
    # frame_tokens: [B,Ht,Wt,D]. Pool latent spatial tokens into teacher grid cells.
    if frame_tokens.ndim != 4:
        raise ValueError(f"frame_tokens must be [B,H,W,D], got {tuple(frame_tokens.shape)}")
    bsz, ht, wt, dim = frame_tokens.shape
    gh, gw = out_grid
    y_edges = torch.linspace(0, ht, gh + 1, device=frame_tokens.device).round().long()
    x_edges = torch.linspace(0, wt, gw + 1, device=frame_tokens.device).round().long()
    cells = []
    for y in range(gh):
        y0 = int(y_edges[y].item())
        y1 = int(y_edges[y + 1].item())
        if y1 <= y0:
            y1 = min(ht, y0 + 1)
        for x in range(gw):
            x0 = int(x_edges[x].item())
            x1 = int(x_edges[x + 1].item())
            if x1 <= x0:
                x1 = min(wt, x0 + 1)
            cells.append(frame_tokens[:, y0:y1, x0:x1, :].reshape(bsz, -1, dim).mean(dim=1))
    return torch.stack(cells, dim=1).view(bsz, gh, gw, dim)


class VideoProcessFlowReadout(nn.Module):
    def __init__(
        self,
        video_dim: int,
        hidden_dim: int = 1024,
        text_dim: int = 4096,
        num_flow_windows: int = 8,
        pooling_type: str = "role_query",
        use_text: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.video_dim = int(video_dim)
        self.hidden_dim = int(hidden_dim)
        self.text_dim = int(text_dim)
        self.num_flow_windows = int(num_flow_windows)
        self.pooling_type = str(pooling_type)
        self.use_text = bool(use_text)
        if self.pooling_type not in {"mean", "role_query"}:
            raise ValueError(f"pooling_type must be mean or role_query, got {pooling_type!r}")
        self.video_norm = nn.LayerNorm(self.video_dim)
        self.video_proj = nn.Linear(self.video_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim) if self.use_text else None
        self.window_embedding = nn.Parameter(torch.randn(self.num_flow_windows, self.hidden_dim) * 0.02)
        if self.pooling_type == "role_query":
            self.role_queries = nn.Parameter(torch.randn(3, self.hidden_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
            self.mlp = MLP(self.hidden_dim, self.hidden_dim, 3, dropout)
        else:
            self.mlp = MLP(self.hidden_dim, self.hidden_dim, self.num_flow_windows * 3 * 3, dropout)

    def forward(
        self,
        video_mid: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        video_tokens_per_frame: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if video_mid.ndim != 3:
            raise ValueError(f"video_mid must be [B,Sv,Dv], got {tuple(video_mid.shape)}")
        h = self.video_proj(self.video_norm(video_mid))
        text = _text_summary(text_context)
        text_h = self.text_proj(text).unsqueeze(1) if (text is not None and self.text_proj is not None) else None
        bsz = video_mid.shape[0]

        if self.pooling_type == "mean":
            outs = []
            for k in range(self.num_flow_windows):
                end = _prefix_end(k, h.shape[1], video_tokens_per_frame)
                pooled = h[:, :end].mean(dim=1)
                if text_h is not None:
                    pooled = pooled + text_h.squeeze(1)
                outs.append(self.mlp(pooled).view(bsz, self.num_flow_windows, 3, 3)[:, k])
            return torch.stack(outs, dim=1), {"video_pooling": "causal_mean", "video_tokens_per_frame": video_tokens_per_frame}

        outs = []
        queries_base = self.role_queries.unsqueeze(0).expand(bsz, -1, -1)
        if text_h is not None:
            queries_base = queries_base + text_h
        for k in range(self.num_flow_windows):
            end = _prefix_end(k, h.shape[1], video_tokens_per_frame)
            role_features, _ = self.cross_attn(queries_base, h[:, :end], h[:, :end], need_weights=False)
            role_features = role_features + self.window_embedding[k].view(1, 1, self.hidden_dim)
            outs.append(self.mlp(role_features))
        out = torch.stack(outs, dim=1)
        return out, {"video_pooling": "causal_role_query", "video_tokens_per_frame": video_tokens_per_frame}


class ActionConditionedProcessFlowReadout(nn.Module):
    def __init__(
        self,
        video_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        text_dim: int = 4096,
        num_flow_windows: int = 8,
        action_segment_len: int = 4,
        pooling_type: str = "role_query",
        use_text: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_flow_windows = int(num_flow_windows)
        self.action_segment_len = int(action_segment_len)
        self.video_readout = VideoProcessFlowReadout(
            video_dim=video_dim,
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            num_flow_windows=num_flow_windows,
            pooling_type=pooling_type,
            use_text=use_text,
            dropout=dropout,
        )
        self.video_role_proj = nn.Linear(3, hidden_dim)
        self.action_norm = nn.LayerNorm(action_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim) if use_text else None
        self.window_embedding = nn.Parameter(torch.randn(self.num_flow_windows, hidden_dim) * 0.02)
        self.mlp = MLP(hidden_dim * 2, hidden_dim, 9, dropout)

    def _pool_action_segments(self, action_mid: torch.Tensor) -> torch.Tensor:
        if action_mid.ndim != 3:
            raise ValueError(f"action_mid must be [B,A,D], got {tuple(action_mid.shape)}")
        bsz, steps, dim = action_mid.shape
        k = self.num_flow_windows
        seg = self.action_segment_len
        need = k * seg
        if steps < need:
            pad = action_mid.new_zeros(bsz, need - steps, dim)
            action_mid = torch.cat([action_mid, pad], dim=1)
        elif steps > need:
            action_mid = action_mid[:, :need]
        return action_mid.view(bsz, k, seg, dim).mean(dim=2)

    def forward(
        self,
        video_mid: torch.Tensor,
        action_mid: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        video_tokens_per_frame: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        vflow, aux = self.video_readout(video_mid, text_context=text_context, video_tokens_per_frame=video_tokens_per_frame)
        video_feat = self.video_role_proj(vflow).mean(dim=2)  # [B,K,H]
        action_seg = self.action_proj(self.action_norm(self._pool_action_segments(action_mid)))
        text = _text_summary(text_context)
        if text is not None and self.text_proj is not None:
            action_seg = action_seg + self.text_proj(text).unsqueeze(1)
        fused = torch.cat([video_feat + self.window_embedding.unsqueeze(0), action_seg], dim=-1)
        out = self.mlp(fused).view(video_mid.shape[0], self.num_flow_windows, 3, 3)
        aux["action_segment_len"] = self.action_segment_len
        return out, aux


class GridFlowReadout(nn.Module):
    def __init__(
        self,
        video_dim: int,
        hidden_dim: int = 1024,
        text_dim: int = 4096,
        num_flow_windows: int = 8,
        grid_size: tuple[int, int] = (8, 8),
        use_text: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.video_dim = int(video_dim)
        self.hidden_dim = int(hidden_dim)
        self.text_dim = int(text_dim)
        self.num_flow_windows = int(num_flow_windows)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        gh, gw = self.grid_size
        self.video_norm = nn.LayerNorm(self.video_dim)
        self.video_proj = nn.Linear(self.video_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim) if use_text else None
        self.window_embedding = nn.Parameter(torch.randn(self.num_flow_windows, self.hidden_dim) * 0.02)
        self.cell_embedding = nn.Parameter(torch.randn(gh * gw, self.hidden_dim) * 0.02)
        self.flow_head = MLP(self.hidden_dim, self.hidden_dim, 3, dropout)
        self.motion_head = MLP(self.hidden_dim, self.hidden_dim, 1, dropout)

    def forward(
        self,
        video_mid: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        video_tokens_per_frame: Optional[int] = None,
        video_grid_size: Optional[tuple[int, int, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if video_mid.ndim != 3:
            raise ValueError(f"video_mid must be [B,Sv,Dv], got {tuple(video_mid.shape)}")
        bsz, seq_len, _ = video_mid.shape
        gh, gw = self.grid_size
        h_video = self.video_proj(self.video_norm(video_mid))
        frames, ht, wt = _resolve_video_grid(seq_len, video_tokens_per_frame, video_grid_size)
        usable = frames * ht * wt
        if usable <= 0 or usable > seq_len:
            raise ValueError(f"Invalid video grid frames={frames} h={ht} w={wt} seq_len={seq_len}")
        if frames < self.num_flow_windows + 1:
            raise ValueError(
                "Grid flow alignment requires one start/end latent pair per window, "
                f"got frames={frames}, num_flow_windows={self.num_flow_windows}"
            )
        h_grid = h_video[:, :usable].view(bsz, frames, ht, wt, self.hidden_dim)

        text = _text_summary(text_context)
        text_h = self.text_proj(text) if (text is not None and self.text_proj is not None) else None
        cell_emb = self.cell_embedding.view(gh, gw, self.hidden_dim)
        per_window = []
        window_frame_pairs = []
        for k in range(self.num_flow_windows):
            start_idx, end_idx = k, k + 1
            start_feat = _pool_spatial_grid(h_grid[:, start_idx], self.grid_size)
            end_feat = _pool_spatial_grid(h_grid[:, end_idx], self.grid_size)
            cell_feat = end_feat - start_feat
            if text_h is not None:
                cell_feat = cell_feat + text_h[:, None, None, :]
            cell_feat = cell_feat + self.window_embedding[k].view(1, 1, 1, self.hidden_dim) + cell_emb.view(1, gh, gw, self.hidden_dim)
            per_window.append(cell_feat)
            window_frame_pairs.append((start_idx, end_idx))
        h = torch.stack(per_window, dim=1)
        flow = self.flow_head(h).view(bsz, self.num_flow_windows, gh, gw, 3)
        motion_logit = self.motion_head(h).view(bsz, self.num_flow_windows, gh, gw)
        return flow, motion_logit, {
            "grid_size": self.grid_size,
            "video_tokens_per_frame": video_tokens_per_frame,
            "video_grid_size": (frames, ht, wt),
            "grid_pooling": "spatial_cell_window_delta",
            "window_frame_pairs": window_frame_pairs,
        }


class ProcessFlowReadout(nn.Module):
    def __init__(
        self,
        enabled: bool = True,
        num_flow_windows: int = 8,
        video_dim: int = 3072,
        action_dim: int = 1024,
        text_dim: int = 4096,
        hidden_dim: int = 1024,
        pooling_type: str = "role_query",
        use_video_readout: bool = True,
        use_action_readout: bool = True,
        action_segment_len: int = 4,
        use_grid_readout: bool = False,
        grid_size: tuple[int, int] = (8, 8),
        dropout: float = 0.0,
        **_: Any,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.use_video_readout = bool(use_video_readout)
        self.use_action_readout = bool(use_action_readout)
        self.use_grid_readout = bool(use_grid_readout)
        self.video = VideoProcessFlowReadout(video_dim, hidden_dim, text_dim, num_flow_windows, pooling_type, True, dropout) if self.use_video_readout else None
        self.action = ActionConditionedProcessFlowReadout(video_dim, action_dim, hidden_dim, text_dim, num_flow_windows, action_segment_len, pooling_type, True, dropout) if self.use_action_readout else None
        self.grid = GridFlowReadout(video_dim, hidden_dim, text_dim, num_flow_windows, grid_size, True, dropout) if self.use_grid_readout else None

    def forward(
        self,
        video_mid: torch.Tensor,
        action_mid: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        sigma_action: Optional[torch.Tensor] = None,
        loss_mask: Optional[dict[str, torch.Tensor]] = None,
        video_tokens_per_frame: Optional[int] = None,
        video_grid_size: Optional[tuple[int, int, int]] = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"pred_vflow": None, "pred_aflow": None, "pred_grid_flow": None, "pred_motion_logit": None, "aux": {}}
        aux: dict[str, Any] = {}
        pred_vflow = None
        pred_aflow = None
        pred_grid_flow = None
        pred_motion_logit = None
        if self.video is not None:
            pred_vflow, aux_v = self.video(video_mid, text_context=text_context, video_tokens_per_frame=video_tokens_per_frame)
            aux["video"] = aux_v
        if self.action is not None and action_mid is not None:
            pred_aflow, aux_a = self.action(video_mid, action_mid, text_context=text_context, video_tokens_per_frame=video_tokens_per_frame)
            aux["action"] = aux_a
        if self.grid is not None:
            pred_grid_flow, pred_motion_logit, aux_g = self.grid(
                video_mid,
                text_context=text_context,
                video_tokens_per_frame=video_tokens_per_frame,
                video_grid_size=video_grid_size,
            )
            aux["grid"] = aux_g
        return {"pred_vflow": pred_vflow, "pred_aflow": pred_aflow, "pred_grid_flow": pred_grid_flow, "pred_motion_logit": pred_motion_logit, "aux": aux}
