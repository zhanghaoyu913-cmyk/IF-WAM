from __future__ import annotations

from typing import Mapping

import torch

LOSS_KEYS = ("video", "action", "vflow", "aflow", "mag", "progress")


def loss_mask_tensor(loss_mask, key: str, like: torch.Tensor | None = None) -> torch.Tensor:
    if key not in LOSS_KEYS:
        raise KeyError(f"Unknown loss mask key: {key}")
    if loss_mask is None:
        value = 1.0
    elif isinstance(loss_mask, Mapping):
        value = loss_mask.get(key, 0.0)
    else:
        raise TypeError(f"loss_mask must be mapping or None, got {type(loss_mask)}")
    if torch.is_tensor(value):
        out = value.float()
    else:
        out = torch.as_tensor(value, dtype=torch.float32)
    if like is not None:
        out = out.to(device=like.device, dtype=like.dtype if like.is_floating_point() else torch.float32)
    return out


def mask_ratio(mask: torch.Tensor | None) -> float:
    if mask is None:
        return 0.0
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().detach().cpu().item())
