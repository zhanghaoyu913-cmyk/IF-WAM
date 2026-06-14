from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

AGENT_IDX = 0
TARGET_IDX = 1
REL_IDX = 2


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return x.mean() if x.numel() else x.sum() * 0.0
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(eps)
    if float(mask.detach().sum().cpu()) <= 0.0:
        return x.sum() * 0.0
    return (x * mask).sum() / denom


def _quality_mask(valid: torch.Tensor, quality: Optional[torch.Tensor]) -> torch.Tensor:
    mask = valid.float()
    if quality is not None:
        q = quality.to(device=valid.device, dtype=mask.dtype)
        while q.ndim < mask.ndim:
            q = q.unsqueeze(-1)
        mask = mask * q
    return mask


def direction_loss(pred, target, valid, quality=None, move_threshold=1e-4):
    eps = 1e-6
    norm_pred = pred.norm(dim=-1)
    norm_gt = target.norm(dim=-1)
    move_mask = (norm_gt > move_threshold).to(dtype=pred.dtype)
    cos = (pred * target).sum(dim=-1) / ((norm_pred + eps) * (norm_gt + eps))
    loss = 1.0 - cos.clamp(-1.0, 1.0)
    return masked_mean(loss, _quality_mask(valid, quality) * move_mask)


def magnitude_loss(pred, target, valid, quality=None, eps=1e-6):
    mag_pred = torch.log(pred.norm(dim=-1) + eps)
    mag_gt = torch.log(target.norm(dim=-1) + eps)
    loss = F.smooth_l1_loss(mag_pred, mag_gt, reduction="none")
    return masked_mean(loss, _quality_mask(valid, quality))


def _relative_scale(target_rel: torch.Tensor, valid_rel: torch.Tensor, mode: str = "sample_max", eps: float = 1e-6) -> torch.Tensor:
    mag = target_rel.detach().norm(dim=-1)
    mask = valid_rel.detach().to(device=target_rel.device, dtype=target_rel.dtype)
    if mode == "sample_mean":
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        scale = (mag * mask).sum(dim=1, keepdim=True) / denom
    elif mode == "batch_max":
        scale = (mag * mask).amax().reshape(1, 1).expand(target_rel.shape[0], 1)
    else:
        scale = (mag * mask).amax(dim=1, keepdim=True)
    return scale.clamp_min(eps).unsqueeze(-1)


def relative_loss(
    pred,
    target,
    valid,
    quality=None,
    normalize: bool = False,
    scale_mode: str = "sample_max",
    normalize_mode: str = "target_scale",
    move_threshold: float = 1e-4,
    eps: float = 1e-6,
):
    pred_rel = pred[:, :, REL_IDX, :]
    target_rel = target[:, :, REL_IDX, :]
    mask = valid[:, :, REL_IDX]
    if quality is not None:
        mask = mask * quality.to(device=mask.device, dtype=mask.dtype)
    if normalize:
        if normalize_mode == "unit_vector":
            move_mask = (target_rel.norm(dim=-1) > move_threshold).to(dtype=mask.dtype)
            mask = mask * move_mask
            pred_rel = F.normalize(pred_rel, dim=-1, eps=eps)
            target_rel = F.normalize(target_rel, dim=-1, eps=eps)
        else:
            scale = _relative_scale(target_rel, mask, mode=scale_mode, eps=eps)
            pred_rel = pred_rel / scale
            target_rel = target_rel / scale
    loss = F.smooth_l1_loss(pred_rel, target_rel, reduction="none").mean(dim=-1)
    return masked_mean(loss, mask)


def static_loss(pred, target, valid, quality=None, static_threshold=1e-4):
    target_vec = target[:, :, TARGET_IDX, :]
    pred_vec = pred[:, :, TARGET_IDX, :]
    static_mask = (target_vec.norm(dim=-1) <= static_threshold).to(dtype=pred.dtype)
    loss = F.smooth_l1_loss(pred_vec, torch.zeros_like(pred_vec), reduction="none").mean(dim=-1)
    mask = valid[:, :, TARGET_IDX] * static_mask
    if quality is not None:
        mask = mask * quality.to(device=mask.device, dtype=mask.dtype)
    return masked_mean(loss, mask)


def consistency_loss(pred, valid=None, quality=None):
    err = pred[:, :, REL_IDX, :] - (pred[:, :, TARGET_IDX, :] - pred[:, :, AGENT_IDX, :])
    loss = err.abs().mean(dim=-1)
    if valid is None:
        mask = torch.ones_like(loss)
    else:
        mask = valid[:, :, AGENT_IDX] * valid[:, :, TARGET_IDX] * valid[:, :, REL_IDX]
    if quality is not None:
        mask = mask * quality.to(device=mask.device, dtype=mask.dtype)
    return masked_mean(loss, mask)


def _sample_mask(loss_mask, key: str, like: torch.Tensor) -> torch.Tensor:
    if loss_mask is None:
        return torch.ones(like.shape[0], device=like.device, dtype=like.dtype)
    value = loss_mask.get(key, 0.0)
    value = value.to(device=like.device, dtype=like.dtype) if torch.is_tensor(value) else torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if value.ndim == 0:
        value = value.expand(like.shape[0])
    return value


def _apply_sample_mask(valid: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    while sample_mask.ndim < valid.ndim:
        sample_mask = sample_mask.unsqueeze(-1)
    return valid * sample_mask


def video_flow_loss(pred_vflow, target_flow, valid_mask, quality, loss_mask, cfg):
    sample = _sample_mask(loss_mask, "vflow", pred_vflow)
    valid = _apply_sample_mask(valid_mask.float(), sample)
    mag_sample = _sample_mask(loss_mask, "mag", pred_vflow)
    mag_valid = _apply_sample_mask(valid_mask.float(), mag_sample)
    vcfg = cfg or {}
    l_dir = direction_loss(pred_vflow, target_flow, valid, quality, _cfg_get(vcfg, "move_threshold", 1e-4))
    l_mag = magnitude_loss(pred_vflow, target_flow, mag_valid, quality) if float(mag_sample.detach().sum().cpu()) > 0 else pred_vflow.sum() * 0.0
    l_rel = relative_loss(
        pred_vflow,
        target_flow,
        valid,
        quality,
        normalize=bool(_cfg_get(vcfg, "normalize_rel", False)),
        scale_mode=str(_cfg_get(vcfg, "rel_scale_mode", "sample_max")),
        eps=float(_cfg_get(vcfg, "rel_scale_eps", 1e-6)),
    )
    l_static = static_loss(pred_vflow, target_flow, valid, quality, _cfg_get(vcfg, "static_threshold", 1e-4))
    l_cons = consistency_loss(pred_vflow, valid, quality)
    total = (_cfg_get(vcfg, "w_dir", 1.0) * l_dir + _cfg_get(vcfg, "w_mag", 0.1) * l_mag + _cfg_get(vcfg, "w_rel", 1.0) * l_rel + _cfg_get(vcfg, "w_static", 0.1) * l_static + _cfg_get(vcfg, "w_consistency", 0.1) * l_cons)
    logs = {"vflow_dir": l_dir, "vflow_mag": l_mag, "vflow_rel": l_rel, "vflow_static": l_static, "vflow_consistency": l_cons}
    return total, logs


def _sigma_window_weight(sigma_action, pred, gamma: float, threshold=None):
    bsz, k = pred.shape[:2]
    if sigma_action is None:
        return torch.ones(bsz, k, device=pred.device, dtype=pred.dtype)
    sigma = sigma_action.to(device=pred.device, dtype=pred.dtype)
    if sigma.ndim == 1:
        sigma = sigma[:, None].expand(bsz, k)
    elif sigma.ndim == 2 and sigma.shape[1] != k:
        seg = max(1, sigma.shape[1] // k)
        sigma = sigma[:, : k * seg].reshape(bsz, k, seg).mean(dim=2)
    weight = (1.0 - sigma).clamp(0, 1) ** gamma
    if threshold is not None:
        weight = weight * (sigma < float(threshold)).to(weight.dtype)
    return weight


def contrastive_consequence_loss(action_feat, target_flow_rel, valid, temperature=0.1):
    if action_feat is None:
        return target_flow_rel.sum() * 0.0
    a = action_feat.reshape(-1, action_feat.shape[-1])
    f = target_flow_rel.reshape(-1, target_flow_rel.shape[-1])
    m = valid.reshape(-1).bool()
    if int(m.sum().detach().cpu()) < 2:
        return target_flow_rel.sum() * 0.0
    a = F.normalize(a[m], dim=-1)
    f = F.normalize(f[m], dim=-1)
    dim = min(a.shape[-1], f.shape[-1])
    logits = a[:, :dim] @ f[:, :dim].T / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def action_flow_loss(pred_aflow, target_flow, valid_mask, quality, loss_mask, sigma_action, cfg):
    sample = _sample_mask(loss_mask, "aflow", pred_aflow)
    valid = _apply_sample_mask(valid_mask.float(), sample)
    acfg = cfg or {}
    sigma_w = _sigma_window_weight(pred=pred_aflow, sigma_action=sigma_action, gamma=_cfg_get(acfg, "sigma_gamma", 2.0), threshold=_cfg_get(acfg, "sigma_threshold", None))
    q = quality * sigma_w if quality is not None else sigma_w
    rel_valid = valid[:, :, REL_IDX]
    agent_valid = valid[:, :, AGENT_IDX]
    l_dir_rel = direction_loss(pred_aflow[:, :, REL_IDX:REL_IDX+1, :], target_flow[:, :, REL_IDX:REL_IDX+1, :], rel_valid.unsqueeze(-1), q)
    l_agent = direction_loss(pred_aflow[:, :, AGENT_IDX:AGENT_IDX+1, :], target_flow[:, :, AGENT_IDX:AGENT_IDX+1, :], agent_valid.unsqueeze(-1), q)
    l_rel = relative_loss(
        pred_aflow,
        target_flow,
        valid,
        q,
        normalize=bool(_cfg_get(acfg, "normalize_rel", False)),
        scale_mode=str(_cfg_get(acfg, "rel_scale_mode", "sample_max")),
        normalize_mode=str(_cfg_get(acfg, "rel_normalize_mode", "unit_vector")),
        move_threshold=float(_cfg_get(acfg, "move_threshold", 1e-4)),
        eps=float(_cfg_get(acfg, "rel_scale_eps", 1e-6)),
    )
    l_cons = consistency_loss(pred_aflow, valid, q)
    l_con = pred_aflow.sum() * 0.0
    total = (_cfg_get(acfg, "w_dir_rel", 1.0) * l_dir_rel + _cfg_get(acfg, "w_agent_dir", 0.5) * l_agent + _cfg_get(acfg, "w_rel", 0.5) * l_rel + _cfg_get(acfg, "w_contrastive", 0.1) * l_con + _cfg_get(acfg, "w_consistency", 0.1) * l_cons)
    logs = {"aflow_dir_rel": l_dir_rel, "aflow_agent_dir": l_agent, "aflow_rel": l_rel, "aflow_contrastive": l_con, "aflow_consistency": l_cons, "aflow_sigma_weight": sigma_w.mean()}
    return total, logs


def flow_score_loss(score_outputs, pred_vflow, pred_aflow, target_flow, valid_mask, quality, loss_mask, cfg):
    if score_outputs is None or pred_vflow is None or pred_aflow is None:
        zero = target_flow.sum() * 0.0
        return zero, {"score": zero}
    score = score_outputs["flow_score"]
    with torch.no_grad():
        cos = F.cosine_similarity(pred_vflow[:, :, REL_IDX, :], pred_aflow[:, :, REL_IDX, :], dim=-1).clamp(-1, 1)
        pseudo = (cos + 1.0) * 0.5
        if score.ndim == 1:
            pseudo = pseudo.mean(dim=1)
    loss = F.smooth_l1_loss(score.float(), pseudo.float(), reduction="none")
    mask = _sample_mask(loss_mask, "progress", pred_vflow)
    if score.ndim == 2:
        mask = mask[:, None].expand_as(score)
    return masked_mean(loss, mask), {"score": masked_mean(loss, mask)}



def grid_flow_loss(pred_grid_flow, pred_motion_logit, target_grid_flow, grid_valid_mask, grid_quality, loss_mask, cfg):
    sample = _sample_mask(loss_mask, "gridflow", pred_grid_flow)
    valid = _apply_sample_mask(grid_valid_mask.float(), sample)
    mag_sample = _sample_mask(loss_mask, "mag", pred_grid_flow)
    mag_valid = _apply_sample_mask(grid_valid_mask.float(), mag_sample)
    gcfg = cfg or {}
    target_norm = target_grid_flow.norm(dim=-1)
    threshold_mode = str(_cfg_get(gcfg, "move_threshold_mode", "fixed"))
    if threshold_mode == "fixed":
        move_threshold = target_norm.new_full(
            target_norm.shape[:2], float(_cfg_get(gcfg, "move_threshold", 1e-4))
        )
    elif threshold_mode == "adaptive_mad":
        flat_norm = target_norm.float().flatten(start_dim=2)
        flat_valid = valid.bool().flatten(start_dim=2)
        nan = torch.full_like(flat_norm, float("nan"))
        masked_norm = torch.where(flat_valid, flat_norm, nan)
        median = torch.nanmedian(masked_norm, dim=-1).values
        abs_dev = torch.abs(flat_norm - median.unsqueeze(-1))
        masked_dev = torch.where(flat_valid, abs_dev, nan)
        mad = torch.nanmedian(masked_dev, dim=-1).values
        threshold_min = float(_cfg_get(gcfg, "move_threshold_min", 1e-3))
        mad_scale = float(_cfg_get(gcfg, "move_threshold_mad_scale", 3.0))
        move_threshold = torch.nan_to_num(median + mad_scale * mad, nan=threshold_min).clamp_min(threshold_min)
        threshold_max = _cfg_get(gcfg, "move_threshold_max", None)
        if threshold_max is not None:
            move_threshold = move_threshold.clamp_max(float(threshold_max))
        move_threshold = move_threshold.to(dtype=target_norm.dtype)
    else:
        raise ValueError(f"Unsupported grid move_threshold_mode={threshold_mode!r}")
    moving_gt = (target_norm > move_threshold[..., None, None]).to(dtype=pred_grid_flow.dtype)
    q = grid_quality.to(device=pred_grid_flow.device, dtype=pred_grid_flow.dtype)
    while q.ndim < valid.ndim:
        q = q.unsqueeze(-1)
    cell_mask = valid * q

    presence_loss = F.binary_cross_entropy_with_logits(pred_motion_logit.float(), moving_gt.float(), reduction="none")
    moving_mask = moving_gt * cell_mask
    static_mask = (1.0 - moving_gt) * cell_mask
    l_presence_pos = masked_mean(presence_loss, moving_mask)
    l_presence_neg = masked_mean(presence_loss, static_mask)
    if bool(_cfg_get(gcfg, "balanced_presence", False)):
        pos_w = float(_cfg_get(gcfg, "presence_pos_weight", 1.0))
        neg_w = float(_cfg_get(gcfg, "presence_neg_weight", 1.0))
        l_presence = pos_w * l_presence_pos + neg_w * l_presence_neg
    else:
        l_presence = masked_mean(presence_loss, cell_mask)

    moving_cell_ratio = masked_mean(moving_gt, cell_mask)
    with torch.no_grad():
        pred_moving = (torch.sigmoid(pred_motion_logit.float()) > float(_cfg_get(gcfg, "presence_threshold", 0.5))).to(dtype=cell_mask.dtype)
    true_pos = pred_moving * moving_mask
    pred_pos = pred_moving * cell_mask
    l_presence_recall = masked_mean(true_pos, moving_mask)
    l_presence_precision = true_pos.sum() / pred_pos.sum().clamp_min(1.0)

    eps = 1e-6
    pred_norm = pred_grid_flow.norm(dim=-1)
    cos = (pred_grid_flow * target_grid_flow).sum(dim=-1) / ((pred_norm + eps) * (target_norm + eps))
    dir_loss = 1.0 - cos.clamp(-1.0, 1.0)
    l_dir = masked_mean(dir_loss, moving_mask)

    if float(mag_sample.detach().sum().cpu()) > 0:
        mag_pred = torch.log(pred_norm + eps)
        mag_gt = torch.log(target_norm + eps)
        l_mag = masked_mean(F.smooth_l1_loss(mag_pred, mag_gt, reduction="none"), mag_valid * q)
    else:
        l_mag = pred_grid_flow.sum() * 0.0

    smoothness_type = str(_cfg_get(gcfg, "smoothness_type", "direction"))
    if smoothness_type == "direction":
        smooth_grid = F.normalize(pred_grid_flow, dim=-1, eps=eps)
        dx = 1.0 - (smooth_grid[:, :, :, 1:, :] * smooth_grid[:, :, :, :-1, :]).sum(dim=-1).clamp(-1.0, 1.0)
        dy = 1.0 - (smooth_grid[:, :, 1:, :, :] * smooth_grid[:, :, :-1, :, :]).sum(dim=-1).clamp(-1.0, 1.0)
    elif smoothness_type == "l1":
        dx = (pred_grid_flow[:, :, :, 1:, :] - pred_grid_flow[:, :, :, :-1, :]).abs().mean(dim=-1)
        dy = (pred_grid_flow[:, :, 1:, :, :] - pred_grid_flow[:, :, :-1, :, :]).abs().mean(dim=-1)
    else:
        raise ValueError(f"Unsupported grid smoothness_type={smoothness_type!r}")

    smoothness_mask_mode = str(_cfg_get(gcfg, "smoothness_mask_mode", "incident_valid"))
    if smoothness_mask_mode == "both_valid":
        mask_x = cell_mask[:, :, :, 1:] * cell_mask[:, :, :, :-1]
        mask_y = cell_mask[:, :, 1:, :] * cell_mask[:, :, :-1, :]
    elif smoothness_mask_mode == "incident_valid":
        # Sparse grid teachers often supervise one cell per window. Regularize
        # edges touching it without inventing ground truth for its neighbors.
        mask_x = torch.maximum(cell_mask[:, :, :, 1:], cell_mask[:, :, :, :-1])
        mask_y = torch.maximum(cell_mask[:, :, 1:, :], cell_mask[:, :, :-1, :])
    else:
        raise ValueError(f"Unsupported grid smoothness_mask_mode={smoothness_mask_mode!r}")
    l_smooth = masked_mean(dx, mask_x) + masked_mean(dy, mask_y)

    total = (
        _cfg_get(gcfg, "w_presence", 1.0) * l_presence
        + _cfg_get(gcfg, "w_dir", 0.5) * l_dir
        + _cfg_get(gcfg, "w_smooth", 0.05) * l_smooth
        + _cfg_get(gcfg, "w_mag", 0.05) * l_mag
    )
    logs = {
        "grid_motion_presence": l_presence,
        "grid_presence_pos": l_presence_pos,
        "grid_presence_neg": l_presence_neg,
        "grid_moving_cell_ratio": moving_cell_ratio,
        "grid_move_threshold": masked_mean(move_threshold, (valid.sum(dim=(-1, -2)) > 0).float()),
        "grid_presence_recall": l_presence_recall,
        "grid_presence_precision": l_presence_precision,
        "grid_dir": l_dir,
        "grid_smooth": l_smooth,
        "grid_mag": l_mag,
    }
    return total, logs
