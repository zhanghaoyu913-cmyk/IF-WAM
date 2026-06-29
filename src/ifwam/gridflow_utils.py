from __future__ import annotations

import numpy as np
import torch


def adaptive_moving_mask_torch(
    flow: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_motion_threshold: float = 1.0e-3,
    mad_scale: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = flow.float().norm(dim=-1)
    valid_bool = valid.bool()
    flat_norm = norm.flatten(start_dim=2)
    flat_valid = valid_bool.flatten(start_dim=2)
    nan = torch.full_like(flat_norm, float("nan"))
    masked_norm = torch.where(flat_valid, flat_norm, nan)
    median = torch.nanmedian(masked_norm, dim=-1).values
    abs_dev = torch.abs(flat_norm - median.unsqueeze(-1))
    masked_dev = torch.where(flat_valid, abs_dev, nan)
    mad = torch.nanmedian(masked_dev, dim=-1).values
    threshold = torch.nan_to_num(
        median + float(mad_scale) * mad,
        nan=float(min_motion_threshold),
    ).clamp_min(float(min_motion_threshold))
    moving = (norm > threshold[..., None, None]) & valid_bool
    return moving.to(dtype=flow.dtype), threshold.to(device=flow.device, dtype=flow.dtype)


def adaptive_moving_mask_numpy(
    flow: np.ndarray,
    valid: np.ndarray,
    *,
    min_motion_threshold: float = 1.0e-3,
    mad_scale: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(flow.astype(np.float32), axis=-1)
    valid_bool = valid.astype(bool)
    masked = np.where(valid_bool, norm, np.nan)
    median = np.nanmedian(masked.reshape(masked.shape[0], -1), axis=1)
    abs_dev = np.abs(norm.reshape(norm.shape[0], -1) - median[:, None])
    masked_dev = np.where(valid_bool.reshape(valid_bool.shape[0], -1), abs_dev, np.nan)
    mad = np.nanmedian(masked_dev, axis=1)
    threshold = median + float(mad_scale) * mad
    threshold = np.where(np.isfinite(threshold), threshold, float(min_motion_threshold))
    threshold = np.maximum(threshold, float(min_motion_threshold))
    moving = valid_bool & (norm > threshold[:, None, None])
    return moving, threshold.astype(np.float32)
