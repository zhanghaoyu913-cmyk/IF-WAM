from typing import Any, Optional, Sequence, Union, Mapping
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger
from ifwam.gridflow_utils import adaptive_moving_mask_torch
from ifwam.models import FlowScoringHead, ProcessFlowReadout
from ifwam.losses.process_flow_losses import action_flow_loss, flow_score_loss, grid_flow_loss, video_flow_loss

from .action_dit import ActionDiT
from .grid_flow_dit import GridFlowDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class GridAuxCrossAttentionDecoder(nn.Module):
    """One-way grid auxiliary decoder.

    Grid tokens are the only tokens updated by this module. Primary video/action
    hidden states enter only as cross-attention memory, so primary forward values
    cannot depend on grid teacher inputs.
    """

    def __init__(
        self,
        grid_dim: int,
        video_dim: int,
        action_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_dim = int(grid_dim)
        self.video_dim = int(video_dim)
        self.action_dim = None if action_dim is None else int(action_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ffn_dim = int(ffn_dim or self.grid_dim * 4)
        if self.grid_dim % self.num_heads != 0:
            raise ValueError(f"grid_dim={self.grid_dim} must be divisible by num_heads={self.num_heads}")
        self.grid_norm = nn.LayerNorm(self.grid_dim)
        self.video_norm = nn.LayerNorm(self.video_dim)
        self.video_proj = nn.Linear(self.video_dim, self.grid_dim)
        self.action_norm = nn.LayerNorm(self.action_dim) if self.action_dim is not None else None
        self.action_proj = nn.Linear(self.action_dim, self.grid_dim) if self.action_dim is not None else None
        self.text_norm = nn.LayerNorm(self.grid_dim)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "self_norm": nn.LayerNorm(self.grid_dim),
                        "cross_norm": nn.LayerNorm(self.grid_dim),
                        "ffn_norm": nn.LayerNorm(self.grid_dim),
                        "self_attn": nn.MultiheadAttention(self.grid_dim, self.num_heads, dropout=dropout, batch_first=True),
                        "cross_attn": nn.MultiheadAttention(self.grid_dim, self.num_heads, dropout=dropout, batch_first=True),
                        "ffn": nn.Sequential(
                            nn.Linear(self.grid_dim, self.ffn_dim),
                            nn.GELU(approximate="tanh"),
                            nn.Dropout(dropout),
                            nn.Linear(self.ffn_dim, self.grid_dim),
                        ),
                    }
                )
            )

    @staticmethod
    def _text_key_padding_mask(text_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if text_mask is None:
            return None
        mask = text_mask
        if mask.ndim == 3:
            mask = mask[:, 0, :]
        if mask.ndim != 2:
            raise ValueError(f"text_mask must be [B,L] or [B,S,L], got {tuple(text_mask.shape)}")
        return ~mask.bool()

    def forward(
        self,
        grid_tokens: torch.Tensor,
        video_hidden: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        action_hidden: Optional[torch.Tensor] = None,
        token_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if grid_tokens.ndim != 3:
            raise ValueError(f"grid_tokens must be [B,S,D], got {tuple(grid_tokens.shape)}")
        if video_hidden.ndim != 3:
            raise ValueError(f"video_hidden must be [B,S,D], got {tuple(video_hidden.shape)}")
        batch_size = grid_tokens.shape[0]
        memory_parts = [self.video_proj(self.video_norm(video_hidden))]
        valid_parts = [torch.ones(batch_size, video_hidden.shape[1], dtype=torch.bool, device=video_hidden.device)]
        if text_context is not None:
            if text_context.ndim != 3:
                raise ValueError(f"text_context must be [B,L,D], got {tuple(text_context.shape)}")
            memory_parts.append(self.text_norm(text_context))
            text_key_padding = self._text_key_padding_mask(text_mask)
            if text_key_padding is None:
                valid_parts.append(torch.ones(batch_size, text_context.shape[1], dtype=torch.bool, device=text_context.device))
            else:
                valid_parts.append(~text_key_padding.to(device=text_context.device))
        if action_hidden is not None:
            if self.action_proj is None or self.action_norm is None:
                raise RuntimeError("action_hidden was provided but action projection is not configured.")
            if action_hidden.ndim != 3:
                raise ValueError(f"action_hidden must be [B,S,D], got {tuple(action_hidden.shape)}")
            memory_parts.append(self.action_proj(self.action_norm(action_hidden)))
            valid_parts.append(torch.ones(batch_size, action_hidden.shape[1], dtype=torch.bool, device=action_hidden.device))
        memory = torch.cat(memory_parts, dim=1)
        memory_valid = torch.cat(valid_parts, dim=1)
        key_padding_mask = ~memory_valid.bool()

        x = self.grid_norm(grid_tokens)
        if token_gate is not None:
            gate = token_gate.to(device=x.device, dtype=x.dtype)
            if gate.ndim != 2 or gate.shape != x.shape[:2]:
                raise ValueError(f"token_gate must be [B,S]={tuple(x.shape[:2])}, got {tuple(gate.shape)}")
            x = x * gate.unsqueeze(-1)
        for layer in self.layers:
            y = layer["self_norm"](x)
            y, _ = layer["self_attn"](y, y, y, need_weights=False)
            x = x + y
            y = layer["cross_norm"](x)
            y, _ = layer["cross_attn"](y, memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + y
            x = x + layer["ffn"](layer["ffn_norm"](x))
            if token_gate is not None:
                x = x * gate.unsqueeze(-1)
        return x


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        ifwam: Optional[Mapping[str, Any]] = None,
        grid_expert: Optional[ActionDiT] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.grid_expert = grid_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.ifwam_cfg = self._normalize_ifwam_cfg(ifwam)
        self.ifwam_enabled = bool(self.ifwam_cfg.get("enabled", False))
        gridfm_cfg = dict(self.ifwam_cfg.get("grid_flow_matching", {}))
        self.grid_flow_matching_enabled = bool(gridfm_cfg.get("enabled", False)) and self.grid_expert is not None
        self.grid_flow_dim = int(gridfm_cfg.get("flow_dim", 3))
        self.grid_flow_num_windows = int(gridfm_cfg.get("num_flow_windows", 2))
        self.grid_flow_grid_size = tuple(int(v) for v in gridfm_cfg.get("grid_size", (8, 8)))
        self.loss_lambda_gridflow_fm = float(gridfm_cfg.get("lambda_gridflow_fm", 0.1))
        self.grid_flow_interaction_mode = str(gridfm_cfg.get("interaction_mode", "legacy"))
        if self.grid_flow_interaction_mode not in {"legacy", "one_way_aux"}:
            raise ValueError(f"Unsupported grid_flow_matching.interaction_mode={self.grid_flow_interaction_mode!r}")
        self.grid_aux_arch = str(gridfm_cfg.get("aux_arch", "shared_mot_legacy_aux"))
        if self.grid_aux_arch not in {"shared_mot_legacy_aux", "cross_attn_decoder"}:
            raise ValueError(f"Unsupported grid_flow_matching.aux_arch={self.grid_aux_arch!r}")
        if self.grid_aux_arch == "cross_attn_decoder" and self.grid_flow_interaction_mode != "one_way_aux":
            raise ValueError("grid_flow_matching.aux_arch=cross_attn_decoder requires interaction_mode=one_way_aux.")
        self.grid_flow_target_mode = str(gridfm_cfg.get("target_mode", "raw_vector_fm"))
        if self.grid_flow_target_mode not in {"raw_vector_fm", "direction_presence"}:
            raise ValueError(f"Unsupported grid_flow_matching.target_mode={self.grid_flow_target_mode!r}")
        self.grid_flow_input_mode = str(gridfm_cfg.get("input_mode", "noisy_teacher"))
        if self.grid_flow_input_mode not in {"noisy_teacher", "learned_query"}:
            raise ValueError(f"Unsupported grid_flow_matching.input_mode={self.grid_flow_input_mode!r}")
        if self.grid_flow_input_mode == "learned_query" and self.grid_flow_target_mode != "direction_presence":
            raise ValueError("grid_flow_matching.input_mode=learned_query requires target_mode=direction_presence.")
        self.grid_quality_application = str(
            gridfm_cfg.get(
                "quality_application",
                "token_and_loss" if bool(gridfm_cfg.get("quality_input_gate", self.grid_flow_interaction_mode == "one_way_aux")) else "loss_only",
            )
        )
        if self.grid_quality_application not in {"loss_only", "token_and_loss", "hard_mask"}:
            raise ValueError(f"Unsupported grid_flow_matching.quality_application={self.grid_quality_application!r}")
        self.grid_context_sources = str(gridfm_cfg.get("context_sources", "video_text_action"))
        if self.grid_context_sources not in {"video_text", "video_text_action"}:
            raise ValueError(f"Unsupported grid_flow_matching.context_sources={self.grid_context_sources!r}")
        self.grid_detach_action_context = bool(gridfm_cfg.get("detach_action_context", False))
        self.grid_shared_grad_scale = float(gridfm_cfg.get("shared_grad_scale", 1.0))
        if self.grid_shared_grad_scale < 0.0:
            raise ValueError("grid_flow_matching.shared_grad_scale must be non-negative.")
        self.grid_quality_hard_threshold = float(gridfm_cfg.get("quality_hard_threshold", 1.0e-6))
        self.grid_quality_input_gate = self.grid_quality_application in {"token_and_loss", "hard_mask"}
        self.grid_min_motion_threshold = float(gridfm_cfg.get("min_motion_threshold", 1e-3))
        self.grid_move_threshold_mad_scale = float(gridfm_cfg.get("move_threshold_mad_scale", 3.0))
        self.grid_presence_loss_weight = float(gridfm_cfg.get("presence_loss_weight", 1.0))
        self.grid_direction_loss_weight = float(gridfm_cfg.get("direction_loss_weight", 1.0))
        self.grid_presence_pos_weight_min = float(gridfm_cfg.get("presence_pos_weight_min", 0.25))
        self.grid_presence_pos_weight_max = float(gridfm_cfg.get("presence_pos_weight_max", 8.0))
        self._warned_grid_magnitude_unreliable = False
        self.grid_aux_decoder = None
        if self.grid_flow_matching_enabled and self.grid_aux_arch == "cross_attn_decoder":
            self.grid_aux_decoder = GridAuxCrossAttentionDecoder(
                grid_dim=int(getattr(self.grid_expert, "hidden_dim")),
                video_dim=int(getattr(self.video_expert, "hidden_dim")),
                action_dim=int(getattr(self.action_expert, "hidden_dim")),
                num_heads=int(gridfm_cfg.get("aux_num_heads", getattr(self.grid_expert, "num_heads", 8))),
                num_layers=int(gridfm_cfg.get("aux_num_layers", 2)),
                ffn_dim=int(gridfm_cfg.get("aux_ffn_dim", getattr(self.grid_expert, "ffn_dim", getattr(self.grid_expert, "hidden_dim") * 4))),
                dropout=float(gridfm_cfg.get("aux_dropout", 0.0)),
            ).to(dtype=torch_dtype)
        self.train_grid_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(gridfm_cfg.get("num_train_timesteps", action_num_train_timesteps)),
            shift=float(gridfm_cfg.get("train_shift", action_train_shift)),
        )
        self.infer_grid_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(gridfm_cfg.get("num_train_timesteps", action_num_train_timesteps)),
            shift=float(gridfm_cfg.get("infer_shift", action_infer_shift)),
        )
        self.process_flow_readout = None
        self.flow_scoring_head = None
        if self.ifwam_enabled and bool(self.ifwam_cfg.get("process_flow_readout", {}).get("enabled", True)):
            readout_cfg = dict(self.ifwam_cfg.get("process_flow_readout", {}))
            readout_cfg.setdefault("video_dim", getattr(self.video_expert, "hidden_dim", 3072))
            readout_cfg.setdefault("action_dim", getattr(self.action_expert, "hidden_dim", 1024))
            self.process_flow_readout = ProcessFlowReadout(**readout_cfg).to(dtype=torch_dtype)
            scoring_cfg = dict(self.ifwam_cfg.get("scoring_head", {}))
            if bool(scoring_cfg.get("enabled", False)):
                scoring_cfg.setdefault("video_dim", readout_cfg.get("video_dim", 3072))
                scoring_cfg.setdefault("action_dim", readout_cfg.get("action_dim", 1024))
                self.flow_scoring_head = FlowScoringHead(**scoring_cfg).to(dtype=torch_dtype)

        self.to(self.device)


    @staticmethod
    def _scale_grid_context_hidden(hidden: torch.Tensor, scale: float, detach: bool = False) -> torch.Tensor:
        if detach or float(scale) == 0.0:
            return hidden.detach()
        if float(scale) == 1.0:
            return hidden
        return hidden.detach() + float(scale) * (hidden - hidden.detach())

    @staticmethod
    def _normalize_ifwam_cfg(ifwam: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        if ifwam is None:
            return {"enabled": False}
        if hasattr(ifwam, "items"):
            out = {}
            for key, value in ifwam.items():
                if hasattr(value, "items"):
                    out[key] = FastWAM._normalize_ifwam_cfg(value)
                else:
                    out[key] = value
            out.setdefault("enabled", False)
            return out
        raise TypeError(f"ifwam config must be mapping-like, got {type(ifwam)}")

    @staticmethod
    def _loss_mask_value(loss_mask, key: str, device: torch.device, dtype: torch.dtype, batch_size: int, default: float = 1.0) -> torch.Tensor:
        if loss_mask is None:
            return torch.full((batch_size,), float(default), device=device, dtype=dtype)
        if not isinstance(loss_mask, Mapping):
            raise TypeError(f"loss_mask must be dict-like, got {type(loss_mask)}")
        value = loss_mask.get(key, default)
        if torch.is_tensor(value):
            out = value.to(device=device, dtype=dtype)
        else:
            out = torch.as_tensor(value, device=device, dtype=dtype)
        if out.ndim == 0:
            out = out.expand(batch_size)
        return out.reshape(batch_size)

    @staticmethod
    def _masked_weighted_mean(per_sample: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = per_sample * weight * mask
        denom = mask.sum().clamp_min(1.0)
        if float(mask.detach().sum().cpu()) <= 0.0:
            return per_sample.sum() * 0.0
        return weighted.sum() / denom

    def _select_ifwam_mid_states(self, mid_states: dict[int, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        mot_cfg = dict(self.ifwam_cfg.get("mot", {}))
        aggregation = str(mot_cfg.get("mid_state_aggregation", "last"))
        if not mid_states:
            raise RuntimeError("IF-WAM enabled but MoT did not return mid_states.")
        layer_items = sorted(mid_states.items(), key=lambda kv: kv[0])
        video_states = [item[1]["video"] for item in layer_items]
        action_states = [item[1]["action"] for item in layer_items]
        if aggregation == "last":
            return video_states[-1], action_states[-1]
        if aggregation == "mean":
            return torch.stack(video_states, dim=0).mean(dim=0), torch.stack(action_states, dim=0).mean(dim=0)
        if aggregation == "concat":
            return torch.cat(video_states, dim=-1), torch.cat(action_states, dim=-1)
        raise ValueError(f"Unsupported IF-WAM mid_state_aggregation={aggregation!r}")

    def _flatten_grid_flow(self, grid_flow: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if grid_flow.ndim != 5:
            raise ValueError(
                f"`grid_flow_teacher` must be [B,K,Gh,Gw,D], got {tuple(grid_flow.shape)}"
            )
        bsz, num_windows, gh, gw, flow_dim = grid_flow.shape
        if int(flow_dim) != int(self.grid_flow_dim):
            raise ValueError(f"grid flow dim mismatch: expected {self.grid_flow_dim}, got {flow_dim}")
        return grid_flow.reshape(bsz, num_windows * gh * gw, flow_dim), (num_windows, gh, gw)

    def _grid_flow_loss_per_sample(
        self,
        pred_grid: torch.Tensor,
        target_grid: torch.Tensor,
        grid_valid: Optional[torch.Tensor],
        grid_quality: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = F.mse_loss(pred_grid.float(), target_grid.float(), reduction="none").mean(dim=-1)
        if grid_valid is None:
            return loss.mean(dim=(1, 2, 3))
        valid = grid_valid.to(device=loss.device, dtype=loss.dtype)
        if grid_quality is not None:
            quality = grid_quality.to(device=loss.device, dtype=loss.dtype)
            if self.grid_quality_application == "hard_mask":
                quality = (quality >= self.grid_quality_hard_threshold).to(dtype=loss.dtype)
            while quality.ndim < valid.ndim:
                quality = quality.unsqueeze(-1)
            valid = valid * quality
        denom = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
        return (loss * valid).sum(dim=(1, 2, 3)) / denom

    def _grid_sample_mask(self, loss_mask, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._loss_mask_value(loss_mask, "gridflow", device, dtype, batch_size, default=1.0)

    def _grid_token_gate(
        self,
        grid_valid: Optional[torch.Tensor],
        grid_quality: Optional[torch.Tensor],
        grid_mask: torch.Tensor,
        grid_shape: tuple[int, int, int],
    ) -> Optional[torch.Tensor]:
        if grid_valid is None:
            return grid_mask[:, None]
        num_windows, gh, gw = grid_shape
        valid = grid_valid.to(device=grid_mask.device, dtype=grid_mask.dtype).reshape(grid_mask.shape[0], num_windows * gh * gw)
        gate = valid * grid_mask[:, None]
        if self.grid_quality_input_gate and grid_quality is not None:
            q = grid_quality.to(device=grid_mask.device, dtype=grid_mask.dtype)
            if self.grid_quality_application == "hard_mask":
                q = (q >= self.grid_quality_hard_threshold).to(dtype=grid_mask.dtype)
            q = q[:, :, None, None].expand(-1, num_windows, gh, gw).reshape(grid_mask.shape[0], num_windows * gh * gw)
            gate = gate * q
        return gate

    def _apply_grid_key_mask(
        self,
        attention_mask: torch.Tensor,
        grid_token_gate: Optional[torch.Tensor],
        video_seq_len: int,
        grid_seq_len: int,
    ) -> torch.Tensor:
        if grid_token_gate is None or grid_seq_len <= 0:
            return attention_mask
        batch_size = grid_token_gate.shape[0]
        mask = attention_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1).clone()
        grid_start = video_seq_len
        grid_end = video_seq_len + grid_seq_len
        valid_key = (grid_token_gate > 0).view(batch_size, 1, 1, grid_seq_len)
        mask[:, :, :, grid_start:grid_end] = mask[:, :, :, grid_start:grid_end] & valid_key
        return mask

    def _grid_moving_mask(
        self,
        flow: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return adaptive_moving_mask_torch(
            flow,
            valid,
            min_motion_threshold=self.grid_min_motion_threshold,
            mad_scale=self.grid_move_threshold_mad_scale,
        )

    def _direction_presence_grid_loss_per_sample(
        self,
        pred_grid: torch.Tensor,
        pred_presence_logit: torch.Tensor,
        target_grid: torch.Tensor,
        clean_grid: torch.Tensor,
        grid_valid: Optional[torch.Tensor],
        grid_quality: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if grid_valid is None:
            valid = torch.ones(clean_grid.shape[:-1], device=clean_grid.device, dtype=clean_grid.dtype)
        else:
            valid = grid_valid.to(device=clean_grid.device, dtype=clean_grid.dtype)
        q = torch.ones(clean_grid.shape[:2], device=clean_grid.device, dtype=clean_grid.dtype) if grid_quality is None else grid_quality.to(device=clean_grid.device, dtype=clean_grid.dtype)
        if self.grid_quality_application == "hard_mask":
            q = (q >= self.grid_quality_hard_threshold).to(dtype=clean_grid.dtype)
        q_cell = q[:, :, None, None].expand_as(valid)
        cell_weight = valid * q_cell
        moving, threshold = self._grid_moving_mask(clean_grid, valid)
        moving_weight = cell_weight * moving.to(dtype=cell_weight.dtype)

        direction_loss = F.mse_loss(pred_grid.float(), target_grid.float(), reduction="none").mean(dim=-1)
        direction_per_sample = (direction_loss * moving_weight).sum(dim=(1, 2, 3)) / moving_weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
        direction_per_sample = torch.where(
            moving_weight.sum(dim=(1, 2, 3)) > 0,
            direction_per_sample,
            direction_loss.sum(dim=(1, 2, 3)) * 0.0,
        )

        target_presence = moving.to(dtype=pred_presence_logit.dtype)
        pos = (target_presence * cell_weight).sum(dim=(1, 2, 3))
        neg = ((1.0 - target_presence) * cell_weight).sum(dim=(1, 2, 3))
        pos_weight = (neg / pos.clamp_min(1.0)).clamp(self.grid_presence_pos_weight_min, self.grid_presence_pos_weight_max)
        bce = F.binary_cross_entropy_with_logits(
            pred_presence_logit.float(),
            target_presence.float(),
            pos_weight=pos_weight[:, None, None, None].to(device=pred_presence_logit.device, dtype=torch.float32),
            reduction="none",
        )
        presence_per_sample = (bce * cell_weight).sum(dim=(1, 2, 3)) / cell_weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
        presence_per_sample = torch.where(
            cell_weight.sum(dim=(1, 2, 3)) > 0,
            presence_per_sample,
            bce.sum(dim=(1, 2, 3)) * 0.0,
        )
        total = self.grid_direction_loss_weight * direction_per_sample + self.grid_presence_loss_weight * presence_per_sample
        logs = {
            "loss_grid_direction": direction_per_sample.mean(),
            "loss_grid_presence": presence_per_sample.mean(),
            "unweighted_direction": direction_per_sample.mean(),
            "unweighted_presence": presence_per_sample.mean(),
            "unweighted_total": total.mean(),
            "grid_moving_cell_ratio": moving_weight.sum() / cell_weight.sum().clamp_min(1.0),
            "grid_move_threshold": threshold.mean(),
            "grid_move_threshold_p10": torch.quantile(threshold.float().flatten(), 0.10),
            "grid_move_threshold_p50": torch.quantile(threshold.float().flatten(), 0.50),
            "grid_move_threshold_p90": torch.quantile(threshold.float().flatten(), 0.90),
            "grid_presence_pos_weight": pos_weight.mean(),
            "_direction_per_sample": direction_per_sample,
            "_presence_per_sample": presence_per_sample,
        }
        return total, logs

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        ifwam: Optional[Mapping[str, Any]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        ifwam_cfg = cls._normalize_ifwam_cfg(ifwam)
        gridfm_cfg = dict(ifwam_cfg.get("grid_flow_matching", {}))
        grid_expert = None
        if bool(gridfm_cfg.get("enabled", False)):
            grid_dit_config = dict(action_dit_config)
            grid_dit_config.pop("action_dim", None)
            grid_dit_config["flow_dim"] = int(gridfm_cfg.get("flow_dim", 3))
            grid_dit_config["num_flow_windows"] = int(gridfm_cfg.get("num_flow_windows", 2))
            grid_dit_config["grid_size"] = tuple(int(v) for v in gridfm_cfg.get("grid_size", (8, 8)))
            grid_dit_config["predict_presence"] = str(gridfm_cfg.get("target_mode", "raw_vector_fm")) == "direction_presence"
            grid_dit_config["learned_query"] = str(gridfm_cfg.get("input_mode", "noisy_teacher")) == "learned_query"
            grid_expert = GridFlowDiT.from_action_pretrained(
                grid_dit_config=grid_dit_config,
                action_dit_pretrained_path=action_dit_pretrained_path,
                skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
                device=device,
                torch_dtype=torch_dtype,
            )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mixtures = {"video": video_expert}
        if grid_expert is not None:
            if int(grid_expert.num_heads) != int(video_expert.num_heads):
                raise ValueError("GridFlow expert `num_heads` must match video expert for MoT mixed attention.")
            if int(grid_expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError("GridFlow expert `attn_head_dim` must match video expert for MoT mixed attention.")
            if int(len(grid_expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError("GridFlow expert `num_layers` must match video expert.")
            mixtures["grid"] = grid_expert
        mixtures["action"] = action_expert
        mot = MoT(
            mixtures=mixtures,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            ifwam=ifwam,
            grid_expert=grid_expert,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "grid_dit_backbone": (
                None if grid_expert is None else ("SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path)
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        grid_seq_len: int = 0,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + grid_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        grid_start = video_seq_len
        action_start = video_seq_len + grid_seq_len
        if grid_seq_len > 0:
            # Grid flow is treated as a video-side causal modality.  Window k
            # represents the transition ending at latent frame k+1, so it must
            # not see video/grid tokens from later latent frames.
            grid_tokens_per_window = max(1, grid_seq_len // max(1, self.grid_flow_num_windows))
            for v_idx in range(video_seq_len):
                video_frame = min(v_idx // video_tokens_per_frame, max(0, self.grid_flow_num_windows))
                visible_grid_windows = min(video_frame, self.grid_flow_num_windows)
                if visible_grid_windows > 0:
                    mask[v_idx, grid_start:grid_start + visible_grid_windows * grid_tokens_per_window] = True
            for g_idx in range(grid_seq_len):
                grid_window = min(g_idx // grid_tokens_per_window, self.grid_flow_num_windows - 1)
                visible_video_tokens = min(video_seq_len, (grid_window + 2) * video_tokens_per_frame)
                visible_grid_tokens = min(grid_seq_len, (grid_window + 1) * grid_tokens_per_window)
                row = grid_start + g_idx
                mask[row, :visible_video_tokens] = True
                mask[row, grid_start:grid_start + visible_grid_tokens] = True
        # action -> action
        mask[action_start:, action_start:] = True
        if grid_seq_len > 0:
            # Grid flow is a training-time auxiliary modality for shaping the
            # video-side representation.  Action tokens condition on aligned
            # video prefixes only; they never attend to grid tokens, so
            # inference does not depend on a teacher-only input.
            action_segment_len = max(1, action_seq_len // max(1, self.grid_flow_num_windows))
            for a_idx in range(action_seq_len):
                action_window = min(a_idx // action_segment_len, self.grid_flow_num_windows - 1)
                visible_video_tokens = min(video_seq_len, (action_window + 2) * video_tokens_per_frame)
                row = action_start + a_idx
                mask[row, :visible_video_tokens] = True
        else:
            # Legacy Fast-WAM action denoising only sees first-frame video.
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            mask[action_start:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        grid_pre = None
        target_grid = None
        grid_valid = None
        grid_quality = None
        timestep_grid = None
        grid_shape = None
        grid_token_gate = None
        grid_mask = None
        grid_clean_for_loss = None
        pred_presence_logit = None
        if self.grid_flow_matching_enabled:
            grid_teacher = sample.get("grid_flow_teacher")
            missing_grid_teacher = grid_teacher is None
            if grid_teacher is None:
                if self.grid_flow_interaction_mode == "legacy":
                    raise ValueError("grid flow matching is enabled but sample has no `grid_flow_teacher`.")
                grid_teacher = torch.zeros(
                    batch_size,
                    self.grid_flow_num_windows,
                    self.grid_flow_grid_size[0],
                    self.grid_flow_grid_size[1],
                    self.grid_flow_dim,
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                if loss_mask := sample.get("loss_mask"):
                    loss_mask["gridflow"] = torch.zeros(batch_size, device=self.device, dtype=self.torch_dtype)
            grid_teacher = grid_teacher.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            loss_mask_for_grid = sample.get("loss_mask")
            grid_mask = self._grid_sample_mask(loss_mask_for_grid, batch_size, self.device, self.torch_dtype)
            if missing_grid_teacher:
                grid_mask = torch.zeros_like(grid_mask)
            grid_tokens_clean, grid_shape = self._flatten_grid_flow(grid_teacher)
            grid_clean_for_loss = grid_teacher
            if self.grid_flow_target_mode == "direction_presence":
                grid_valid_tmp = sample.get("grid_flow_valid")
                grid_valid_for_target = (
                    torch.ones(grid_teacher.shape[:-1], device=self.device, dtype=self.torch_dtype)
                    if grid_valid_tmp is None
                    else grid_valid_tmp.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                )
                moving_tmp, _ = self._grid_moving_mask(grid_teacher, grid_valid_for_target)
                grid_direction = F.normalize(grid_teacher.float(), dim=-1, eps=1e-6).to(dtype=self.torch_dtype)
                grid_direction = grid_direction * moving_tmp.unsqueeze(-1).to(dtype=self.torch_dtype)
                grid_tokens_clean, _ = self._flatten_grid_flow(grid_direction)
            if self.grid_flow_input_mode == "learned_query":
                timestep_grid = torch.zeros(batch_size, device=self.device, dtype=grid_tokens_clean.dtype)
                noisy_grid = torch.zeros_like(grid_tokens_clean)
                target_grid = grid_tokens_clean
            else:
                noise_grid = torch.randn_like(grid_tokens_clean)
                timestep_grid = self.train_grid_scheduler.sample_training_t(
                    batch_size=batch_size,
                    device=self.device,
                    dtype=grid_tokens_clean.dtype,
                )
                noisy_grid = self.train_grid_scheduler.add_noise(grid_tokens_clean, noise_grid, timestep_grid)
                target_grid = self.train_grid_scheduler.training_target(grid_tokens_clean, noise_grid, timestep_grid)
            grid_valid = sample.get("grid_flow_valid")
            if grid_valid is not None:
                grid_valid = grid_valid.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            grid_quality = sample.get("grid_flow_quality")
            if grid_quality is not None:
                grid_quality = grid_quality.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            grid_token_gate = self._grid_token_gate(grid_valid, grid_quality, grid_mask, grid_shape)
            if self.grid_flow_target_mode == "raw_vector_fm" and not self._warned_grid_magnitude_unreliable:
                warnings.warn(
                    "GridFM raw_vector_fm uses vector magnitude. Existing grid-flow metadata marks "
                    "magnitude_reliable=false; prefer target_mode=direction_presence for teacher-consistent training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_grid_magnitude_unreliable = True

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        if self.grid_flow_matching_enabled:
            if self.grid_flow_interaction_mode == "legacy" or grid_mask is None or bool((grid_mask > 0).any().detach().item()):
                grid_pre = self.grid_expert.pre_dit(
                    grid_tokens=noisy_grid,
                    timestep=timestep_grid,
                    context=context,
                    context_mask=context_mask,
                    token_gate=grid_token_gate if self.grid_flow_interaction_mode == "one_way_aux" else None,
                    use_learned_query=self.grid_flow_input_mode == "learned_query",
                )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        grid_tokens = grid_pre["tokens"] if grid_pre is not None else None

        mot_cfg = dict(self.ifwam_cfg.get("mot", {})) if self.ifwam_enabled else {}
        return_mid_states = bool(mot_cfg.get("return_mid_states", False)) and self.process_flow_readout is not None
        primary_attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        primary_embeds_all = {
            "video": video_tokens,
            "action": action_tokens,
        }
        primary_freqs_all = {
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        }
        primary_context_all = {
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        }
        primary_t_mod_all = {
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        }

        tokens_out = None
        mid_states = None
        if grid_tokens is not None and self.grid_flow_interaction_mode != "one_way_aux":
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_tokens.shape[1],
                action_seq_len=action_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_tokens.device,
                grid_seq_len=grid_tokens.shape[1],
            )
            embeds_all = {"video": video_tokens}
            freqs_all = {"video": video_pre["freqs"]}
            context_all = {
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
            }
            t_mod_all = {"video": video_pre["t_mod"]}
            embeds_all["grid"] = grid_tokens
            freqs_all["grid"] = grid_pre["freqs"]
            context_all["grid"] = {
                "context": grid_pre["context"],
                "mask": grid_pre["context_mask"],
            }
            t_mod_all["grid"] = grid_pre["t_mod"]
            embeds_all["action"] = action_tokens
            freqs_all["action"] = action_pre["freqs"]
            context_all["action"] = {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            }
            t_mod_all["action"] = action_pre["t_mod"]
            mot_out = self.mot(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
                return_mid_states=return_mid_states,
                mid_layer_indices=mot_cfg.get("mid_layer_indices", None),
            )
            if return_mid_states:
                tokens_out, mid_states = mot_out
            else:
                tokens_out, mid_states = mot_out, None
        else:
            mot_out = self.mot(
                embeds_all=primary_embeds_all,
                attention_mask=primary_attention_mask,
                freqs_all=primary_freqs_all,
                context_all=primary_context_all,
                t_mod_all=primary_t_mod_all,
                return_mid_states=return_mid_states,
                mid_layer_indices=mot_cfg.get("mid_layer_indices", None),
            )
            if return_mid_states:
                tokens_out, mid_states = mot_out
            else:
                tokens_out, mid_states = mot_out, None

            if grid_tokens is not None and self.grid_flow_interaction_mode == "one_way_aux":
                video_seq_len = tokens_out["video"].shape[1]
                grid_seq_len = grid_tokens.shape[1]
                video_for_grid = self._scale_grid_context_hidden(
                    tokens_out["video"],
                    scale=self.grid_shared_grad_scale,
                    detach=False,
                )
                if self.grid_aux_arch == "cross_attn_decoder":
                    if self.grid_aux_decoder is None:
                        raise RuntimeError("grid_aux_decoder is required for aux_arch=cross_attn_decoder.")
                    action_for_grid = None
                    if self.grid_context_sources == "video_text_action":
                        action_for_grid = self._scale_grid_context_hidden(
                            tokens_out["action"],
                            scale=self.grid_shared_grad_scale,
                            detach=self.grid_detach_action_context,
                        )
                    grid_decoded = self.grid_aux_decoder(
                        grid_tokens=grid_tokens,
                        video_hidden=video_for_grid,
                        text_context=grid_pre["context"],
                        text_mask=grid_pre["context_mask"],
                        action_hidden=action_for_grid,
                        token_gate=grid_token_gate,
                    )
                    tokens_out = dict(tokens_out)
                    tokens_out["grid"] = grid_decoded
                else:
                    embeds_all = {
                        "video": video_for_grid,
                        "grid": grid_tokens,
                    }
                    freqs_all = {
                        "video": video_pre["freqs"],
                        "grid": grid_pre["freqs"],
                    }
                    context_all = {
                        "video": {
                            "context": self._scale_grid_context_hidden(
                                video_pre["context"],
                                scale=self.grid_shared_grad_scale,
                                detach=False,
                            ),
                            "mask": video_pre["context_mask"],
                        },
                        "grid": {
                            "context": grid_pre["context"],
                            "mask": grid_pre["context_mask"],
                        },
                    }
                    t_mod_all = {
                        "video": self._scale_grid_context_hidden(
                            video_pre["t_mod"],
                            scale=self.grid_shared_grad_scale,
                            detach=False,
                        ),
                        "grid": grid_pre["t_mod"],
                    }
                    if self.grid_context_sources == "video_text_action":
                        action_for_grid = self._scale_grid_context_hidden(
                            tokens_out["action"],
                            scale=self.grid_shared_grad_scale,
                            detach=self.grid_detach_action_context,
                        )
                        embeds_all["action"] = action_for_grid
                        freqs_all["action"] = action_pre["freqs"]
                        context_all["action"] = {
                            "context": self._scale_grid_context_hidden(
                                action_pre["context"],
                                scale=self.grid_shared_grad_scale,
                                detach=self.grid_detach_action_context,
                            ),
                            "mask": action_pre["context_mask"],
                        }
                        t_mod_all["action"] = self._scale_grid_context_hidden(
                            action_pre["t_mod"],
                            scale=self.grid_shared_grad_scale,
                            detach=self.grid_detach_action_context,
                        )
                        aux_attention_mask = self._build_mot_attention_mask(
                            video_seq_len=video_seq_len,
                            action_seq_len=tokens_out["action"].shape[1],
                            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                            device=video_tokens.device,
                            grid_seq_len=grid_seq_len,
                        )
                        action_start = video_seq_len + grid_seq_len
                        aux_attention_mask[:video_seq_len, video_seq_len:action_start] = False
                        aux_attention_mask[action_start:, video_seq_len:action_start] = False
                        aux_attention_mask[video_seq_len:action_start, :video_seq_len] = True
                        aux_attention_mask[video_seq_len:action_start, action_start:] = True
                        aux_attention_mask = self._apply_grid_key_mask(
                            attention_mask=aux_attention_mask,
                            grid_token_gate=grid_token_gate,
                            video_seq_len=video_seq_len,
                            grid_seq_len=grid_seq_len,
                        )
                    else:
                        aux_attention_mask = torch.ones(
                            (video_seq_len + grid_seq_len, video_seq_len + grid_seq_len),
                            dtype=torch.bool,
                            device=video_tokens.device,
                        )
                        aux_attention_mask[:video_seq_len, video_seq_len:] = False
                        aux_attention_mask = self._apply_grid_key_mask(
                            attention_mask=aux_attention_mask,
                            grid_token_gate=grid_token_gate,
                            video_seq_len=video_seq_len,
                            grid_seq_len=grid_seq_len,
                        )
                    aux_out = self.mot(
                        embeds_all=embeds_all,
                        attention_mask=aux_attention_mask,
                        freqs_all=freqs_all,
                        context_all=context_all,
                        t_mod_all=t_mod_all,
                        return_mid_states=False,
                    )
                    tokens_out = dict(tokens_out)
                    tokens_out["grid"] = aux_out["grid"]

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_grid = None
        if grid_pre is not None:
            pred_grid = self.grid_expert.post_dit(tokens_out["grid"], grid_pre)
            if self.grid_flow_target_mode == "direction_presence":
                pred_presence_logit = self.grid_expert.post_presence(tokens_out["grid"]).view(
                    batch_size,
                    self.grid_flow_num_windows,
                    self.grid_flow_grid_size[0],
                    self.grid_flow_grid_size[1],
                )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_mask = sample.get("loss_mask")
        video_mask = self._loss_mask_value(loss_mask, "video", loss_video_per_sample.device, loss_video_per_sample.dtype, batch_size, default=1.0)
        loss_video = self._masked_weighted_mean(loss_video_per_sample, video_weight, video_mask)

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        action_mask = self._loss_mask_value(loss_mask, "action", action_loss_per_sample.device, action_loss_per_sample.dtype, batch_size, default=1.0)
        loss_action = self._masked_weighted_mean(action_loss_per_sample, action_weight, action_mask)

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "unweighted_action_loss": float(loss_action.detach().item()),
            "weighted_action_loss": self.loss_lambda_action * float(loss_action.detach().item()),
            "action_prediction_norm": float(pred_action.float().norm(dim=-1).mean().detach().item()),
            "action_target_norm": float(target_action.float().norm(dim=-1).mean().detach().item()),
            "ifwam/loss_mask_video_ratio": float(video_mask.float().mean().detach().item()),
            "ifwam/loss_mask_action_ratio": float(action_mask.float().mean().detach().item()),
        }
        if pred_grid is not None and target_grid is not None and grid_shape is not None:
            num_windows, gh, gw = grid_shape
            pred_grid_view = pred_grid.view(batch_size, num_windows, gh, gw, self.grid_flow_dim)
            target_grid_view = target_grid.view(batch_size, num_windows, gh, gw, self.grid_flow_dim)
            if self.grid_flow_target_mode == "direction_presence":
                if pred_presence_logit is None or grid_clean_for_loss is None:
                    raise RuntimeError("direction_presence mode requires presence logits and clean grid target.")
                grid_loss_per_sample, grid_logs = self._direction_presence_grid_loss_per_sample(
                    pred_grid=pred_grid_view,
                    pred_presence_logit=pred_presence_logit,
                    target_grid=target_grid_view,
                    clean_grid=grid_clean_for_loss,
                    grid_valid=grid_valid,
                    grid_quality=grid_quality,
                )
            else:
                grid_loss_per_sample = self._grid_flow_loss_per_sample(
                    pred_grid=pred_grid_view,
                    target_grid=target_grid_view,
                    grid_valid=grid_valid,
                    grid_quality=grid_quality,
                )
                grid_logs = {}
            if self.grid_flow_input_mode == "learned_query":
                grid_weight = torch.ones_like(grid_loss_per_sample)
            else:
                grid_weight = self.train_grid_scheduler.training_weight(timestep_grid).to(
                    grid_loss_per_sample.device, dtype=grid_loss_per_sample.dtype
                )
            grid_mask = self._loss_mask_value(loss_mask, "gridflow", grid_loss_per_sample.device, grid_loss_per_sample.dtype, batch_size, default=1.0)
            loss_grid_fm = self._masked_weighted_mean(grid_loss_per_sample, grid_weight, grid_mask)
            loss_total = loss_total + self.loss_lambda_gridflow_fm * loss_grid_fm
            loss_dict["loss_gridflow_fm"] = float(loss_grid_fm.detach().item())
            loss_dict["loss_grid_raw"] = float(loss_grid_fm.detach().item())
            loss_dict["loss_grid_scaled"] = self.loss_lambda_gridflow_fm * float(loss_grid_fm.detach().item())
            loss_dict["lambda_scaled_total"] = self.loss_lambda_gridflow_fm * float(loss_grid_fm.detach().item())
            loss_dict["unweighted_gridflow_fm_loss"] = float(loss_grid_fm.detach().item())
            loss_dict["weighted_gridflow_fm_loss"] = self.loss_lambda_gridflow_fm * float(loss_grid_fm.detach().item())
            loss_dict["ifwam/loss_mask_gridflow_ratio"] = float(grid_mask.float().mean().detach().item())
            loss_dict["grid_lambda_effective"] = float(self.loss_lambda_gridflow_fm)
            if grid_valid is not None:
                valid_float = grid_valid.float()
                loss_dict["grid_valid_cell_ratio"] = float(valid_float.mean().detach().item())
                loss_dict["num_grid_tokens_valid"] = float(valid_float.sum().detach().item())
            if grid_quality is not None:
                loss_dict["grid_quality_mean"] = float(grid_quality.float().mean().detach().item())
            if grid_clean_for_loss is not None and grid_valid is not None:
                moving_tmp, _ = self._grid_moving_mask(grid_clean_for_loss, grid_valid)
                norm = grid_clean_for_loss.float().norm(dim=-1)
                moving_norm = norm[moving_tmp.bool()]
                loss_dict["grid_moving_cell_ratio"] = float(moving_tmp.float().mean().detach().item())
                loss_dict["grid_flow_norm_mean_moving"] = float(moving_norm.mean().detach().item()) if moving_norm.numel() else 0.0
                loss_dict["grid_flow_norm_p50_moving"] = float(moving_norm.median().detach().item()) if moving_norm.numel() else 0.0
            loss_dict["grid_teacher_sample_coverage"] = float((grid_mask > 0).float().mean().detach().item())
            if "_direction_per_sample" in grid_logs:
                dps = grid_logs["_direction_per_sample"].to(device=grid_loss_per_sample.device, dtype=grid_loss_per_sample.dtype)
                pps = grid_logs["_presence_per_sample"].to(device=grid_loss_per_sample.device, dtype=grid_loss_per_sample.dtype)
                loss_dict["weighted_direction"] = float(self._masked_weighted_mean(dps, grid_weight, grid_mask).detach().item())
                loss_dict["weighted_presence"] = float(self._masked_weighted_mean(pps, grid_weight, grid_mask).detach().item())
                loss_dict["weighted_total"] = float(loss_grid_fm.detach().item())
            for k, v in grid_logs.items():
                if k.startswith("_"):
                    continue
                loss_dict[k] = float(v.detach().item())

        if self.ifwam_enabled and self.process_flow_readout is not None and mid_states is not None:
            target_flow = sample.get("iflow_teacher")
            valid_flow = sample.get("iflow_valid")
            quality = sample.get("iflow_quality")
            if target_flow is not None and valid_flow is not None and quality is not None:
                target_flow = target_flow.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                valid_flow = valid_flow.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                quality = quality.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                video_mid, action_mid = self._select_ifwam_mid_states(mid_states)
                flow_outputs = self.process_flow_readout(
                    video_mid=video_mid,
                    action_mid=action_mid,
                    text_context=context,
                    sigma_action=timestep_action,
                    loss_mask=loss_mask,
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    video_grid_size=tuple(video_pre["meta"].get("grid_size", (0, 0, 0))),
                )
                losses_cfg = dict(self.ifwam_cfg.get("losses", {}))
                pred_vflow = flow_outputs.get("pred_vflow")
                pred_aflow = flow_outputs.get("pred_aflow")
                pred_grid_flow = flow_outputs.get("pred_grid_flow")
                pred_motion_logit = flow_outputs.get("pred_motion_logit")
                if pred_vflow is not None:
                    l_vflow, logs = video_flow_loss(pred_vflow, target_flow, valid_flow, quality, loss_mask, losses_cfg.get("vflow", {}))
                    loss_total = loss_total + float(losses_cfg.get("lambda_vflow", 0.0)) * l_vflow
                    loss_dict["loss_vflow"] = float(l_vflow.detach().item())
                    for k, v in logs.items():
                        loss_dict[f"ifwam/{k}"] = float(v.detach().item())
                if pred_aflow is not None:
                    # `sample_training_t` returns scheduler timesteps in [0, num_train_timesteps],
                    # while action_flow_loss expects normalized noise sigma in [0, 1].
                    sigma_action = timestep_action / float(self.train_action_scheduler.num_train_timesteps)
                    l_aflow, logs = action_flow_loss(pred_aflow, target_flow, valid_flow, quality, loss_mask, sigma_action, losses_cfg.get("aflow", {}))
                    loss_total = loss_total + float(losses_cfg.get("lambda_aflow", 0.0)) * l_aflow
                    loss_dict["loss_aflow"] = float(l_aflow.detach().item())
                    for k, v in logs.items():
                        loss_dict[f"ifwam/{k}"] = float(v.detach().item())
                grid_target = sample.get("grid_flow_teacher")
                grid_valid = sample.get("grid_flow_valid")
                grid_quality = sample.get("grid_flow_quality")
                if pred_grid_flow is not None and pred_motion_logit is not None and grid_target is not None and grid_valid is not None and grid_quality is not None:
                    grid_target = grid_target.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                    grid_valid = grid_valid.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                    grid_quality = grid_quality.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                    l_grid, logs = grid_flow_loss(pred_grid_flow, pred_motion_logit, grid_target, grid_valid, grid_quality, loss_mask, losses_cfg.get("gridflow", {}))
                    grid_lambda = float(losses_cfg.get("lambda_gridflow", 0.0))
                    loss_total = loss_total + grid_lambda * l_grid
                    loss_dict["loss_gridflow"] = float(l_grid.detach().item())
                    loss_dict["unweighted_gridflow_loss"] = float(l_grid.detach().item())
                    loss_dict["weighted_gridflow_loss"] = grid_lambda * float(l_grid.detach().item())
                    for k, v in logs.items():
                        loss_dict[f"ifwam/{k}"] = float(v.detach().item())
                if self.flow_scoring_head is not None and pred_vflow is not None and pred_aflow is not None:
                    score_outputs = self.flow_scoring_head(video_mid=video_mid, action_mid=action_mid, pred_vflow=pred_vflow, pred_aflow=pred_aflow, text_context=context)
                    l_score, logs = flow_score_loss(score_outputs, pred_vflow, pred_aflow, target_flow, valid_flow, quality, loss_mask, losses_cfg.get("score", {}))
                    loss_total = loss_total + float(losses_cfg.get("lambda_score", 0.0)) * l_score
                    loss_dict["loss_score"] = float(l_score.detach().item())
                loss_dict["ifwam/flow_valid_ratio"] = float((valid_flow > 0).float().mean().detach().item())
                loss_dict["ifwam/flow_quality_mean"] = float(quality.float().mean().detach().item())
                if isinstance(loss_mask, Mapping):
                    for key in ("video", "action", "vflow", "aflow", "gridflow", "mag"):
                        value = loss_mask.get(key)
                        if torch.is_tensor(value):
                            loss_dict[f"ifwam/loss_mask_{key}_ratio"] = float(value.float().mean().detach().item())

        loss_dict["total_loss"] = float(loss_total.detach().item())
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.process_flow_readout is not None:
            payload["process_flow_readout"] = self.process_flow_readout.state_dict()
        if self.flow_scoring_head is not None:
            payload["flow_scoring_head"] = self.flow_scoring_head.state_dict()
        if self.grid_aux_decoder is not None:
            payload["grid_aux_decoder"] = self.grid_aux_decoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
        if self.process_flow_readout is not None:
            if "process_flow_readout" in payload:
                self.process_flow_readout.load_state_dict(payload["process_flow_readout"], strict=True)
            else:
                logger.warning("Checkpoint has no `process_flow_readout` weights; keeping current initialization.")
        elif "process_flow_readout" in payload:
            logger.warning("Checkpoint contains process-flow readout weights but IF-WAM readout is disabled; ignoring.")
        if self.flow_scoring_head is not None:
            if "flow_scoring_head" in payload:
                self.flow_scoring_head.load_state_dict(payload["flow_scoring_head"], strict=True)
            else:
                logger.warning("Checkpoint has no `flow_scoring_head` weights; keeping current initialization.")
        elif "flow_scoring_head" in payload:
            logger.warning("Checkpoint contains flow-scoring weights but scoring head is disabled; ignoring.")
        if self.grid_aux_decoder is not None:
            if "grid_aux_decoder" in payload:
                self.grid_aux_decoder.load_state_dict(payload["grid_aux_decoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `grid_aux_decoder` weights; keeping current initialization.")
        elif "grid_aux_decoder" in payload:
            logger.warning("Checkpoint contains grid aux decoder weights but current model has no grid_aux_decoder; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
