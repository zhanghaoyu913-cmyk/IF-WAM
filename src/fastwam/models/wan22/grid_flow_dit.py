import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import DiTBlock, precompute_freqs_cis, sinusoidal_embedding_1d

logger = get_logger(__name__)


class GridFlowDiT(nn.Module):
    """Continuous-token DiT expert for grid-flow denoising.

    The transformer backbone intentionally matches ActionDiT so it can reuse the
    existing action backbone checkpoint. Grid-specific input/output projections
    and grid position embeddings stay randomly initialized and are learned during
    GridFM training.
    """

    GRID_BACKBONE_SKIP_PREFIXES = ("grid_encoder.", "grid_head.", "grid_position_embedding")
    GRID_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        flow_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        num_flow_windows: int = 2,
        grid_size: tuple[int, int] | list[int] = (8, 8),
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.flow_dim = int(flow_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_flow_windows = int(num_flow_windows)
        self.grid_size = tuple(int(v) for v in grid_size)

        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {self.num_heads}")
        if self.attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {self.attn_head_dim}")
        if self.attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {self.attn_head_dim}")
        if self.num_flow_windows <= 0:
            raise ValueError(f"`num_flow_windows` must be > 0, got {self.num_flow_windows}")
        if len(self.grid_size) != 2 or self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError(f"`grid_size` must be [Gh, Gw] with positive values, got {self.grid_size}")

        self.grid_encoder = nn.Linear(self.flow_dim, self.hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=eps,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.grid_head = nn.Linear(self.hidden_dim, self.flow_dim)
        grid_seq_len = self.num_flow_windows * self.grid_size[0] * self.grid_size[1]
        self.grid_position_embedding = nn.Parameter(torch.zeros(1, grid_seq_len, self.hidden_dim))
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=1024)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.GRID_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_action_pretrained(
        cls,
        grid_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "GridFlowDiT":
        if grid_dit_config is None:
            raise ValueError("`grid_dit_config` is required for GridFlowDiT.from_action_pretrained().")
        cfg = dict(grid_dit_config)
        if "flow_dim" not in cfg and "action_dim" in cfg:
            cfg["flow_dim"] = cfg.pop("action_dim")
        if "flow_dim" not in cfg:
            raise ValueError("`grid_dit_config.flow_dim` is required for GridFlowDiT.")

        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping GridFlowDiT action-backbone load (`skip_dit_load_from_pretrain=True`); "
                "initializing grid expert randomly and expecting checkpoint override."
            )
            return cls(**cfg).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No action backbone path provided, initializing GridFlowDiT with random weights.")
            return cls(**cfg).to(device=device, dtype=torch_dtype)

        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        grid_expert = cls(**cfg).to(device=device, dtype=torch_dtype)
        grid_state = grid_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(grid_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {action_dit_pretrained_path}")
        expected_meta = {
            "hidden_dim": int(cfg["hidden_dim"]),
            "ffn_dim": int(cfg["ffn_dim"]),
            "num_layers": int(cfg["num_layers"]),
            "num_heads": int(cfg["num_heads"]),
            "attn_head_dim": int(cfg["attn_head_dim"]),
            "text_dim": int(cfg["text_dim"]),
            "freq_dim": int(cfg["freq_dim"]),
            "eps": float(cfg["eps"]),
        }
        for key in cls.GRID_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )
        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "GridFlowDiT action-backbone key mismatch. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(grid_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        grid_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded GridFlowDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.GRID_BACKBONE_SKIP_PREFIXES),
        )
        return grid_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        grid_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if grid_tokens.ndim != 3:
            raise ValueError(f"`grid_tokens` must be 3D [B, S, flow_dim], got shape {tuple(grid_tokens.shape)}")
        if grid_tokens.shape[2] != self.flow_dim:
            raise ValueError(f"`grid_tokens` last dim must be {self.flow_dim}, got {grid_tokens.shape[2]}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")

        batch_size, seq_len, _ = grid_tokens.shape
        if seq_len > self.grid_position_embedding.shape[1]:
            raise ValueError(
                f"Grid token length {seq_len} exceeds configured grid sequence "
                f"{self.grid_position_embedding.shape[1]}."
            )
        if seq_len > self.freqs.shape[0]:
            raise ValueError(f"Grid token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}.")
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between grid tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, grid timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], "
                    f"got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        tokens = self.grid_encoder(grid_tokens)
        tokens = tokens + self.grid_position_embedding[:, :seq_len].to(device=tokens.device, dtype=tokens.dtype)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_flow_windows": self.num_flow_windows,
                "grid_size": self.grid_size,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        del pre_state
        return self.grid_head(tokens)

    def forward(
        self,
        grid_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            grid_tokens=grid_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)
        return self.post_dit(x, pre_state)
