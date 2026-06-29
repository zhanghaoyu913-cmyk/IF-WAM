#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fastwam.runtime import _mixed_precision_to_model_dtype
from fastwam.utils.run_artifacts import write_run_artifacts


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = {kk: vv.clone() if torch.is_tensor(vv) else copy.deepcopy(vv) for kk, vv in v.items()}
        else:
            out[k] = copy.deepcopy(v)
    return out


def _context_variant(batch: dict[str, Any], variant: str) -> dict[str, Any]:
    out = _clone_batch(batch)
    if "context" not in out:
        return out
    if variant == "correct":
        return out
    if variant == "shuffled":
        out["context"] = out["context"].flip(0)
        if "context_mask" in out and torch.is_tensor(out["context_mask"]):
            out["context_mask"] = out["context_mask"].flip(0)
        return out
    if variant == "zero":
        out["context"] = torch.zeros_like(out["context"])
        return out
    raise ValueError(variant)


def _norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm().item()) ** 2
    return total ** 0.5


def _flat_grads(params, limit: int = 2_000_000):
    pieces = []
    n = 0
    names = []
    for name, p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float().flatten()
        take = min(g.numel(), max(0, limit - n))
        if take <= 0:
            break
        pieces.append(g[:take].cpu())
        names.append(name)
        n += take
    if not pieces:
        return None, names
    return torch.cat(pieces), names


def _binary_metrics(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    mask = valid.bool()
    if mask.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auprc": 0.0}
    probs = torch.sigmoid(logits.detach().float())[mask]
    y = target.detach().bool()[mask]
    pred = probs >= 0.5
    tp = (pred & y).sum().float()
    fp = (pred & ~y).sum().float()
    fn = (~pred & y).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    order = torch.argsort(probs, descending=True)
    y_sorted = y.float()[order]
    tp_cum = torch.cumsum(y_sorted, dim=0)
    denom = torch.arange(1, y_sorted.numel() + 1, device=y_sorted.device, dtype=torch.float32)
    precision_curve = tp_cum / denom
    positives = y_sorted.sum().clamp_min(1.0)
    auprc = (precision_curve * y_sorted).sum() / positives
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "auprc": float(auprc)}


def _patch_timestep_samplers(model, t: float):
    originals = []
    for scheduler in (model.train_video_scheduler, model.train_action_scheduler, model.train_grid_scheduler):
        orig = scheduler.sample_training_t
        originals.append((scheduler, orig))

        def fixed(batch_size, device, dtype, _t=float(t)):
            return torch.full((batch_size,), _t, device=device, dtype=dtype)

        scheduler.sample_training_t = fixed
    return originals


def _restore_timestep_samplers(originals):
    for scheduler, orig in originals:
        scheduler.sample_training_t = orig


def _selected_shared_params(model):
    items = list(model.mot.named_parameters())
    if not items:
        return []
    picks = [items[0], items[len(items) // 2], items[-1]]
    seen = set()
    out = []
    for name, p in picks:
        if id(p) not in seen:
            out.append((f"mot.{name}", p))
            seen.add(id(p))
    return out


def _run_variant(model, batch: dict[str, Any], seed: int, timestep: float):
    captured = {}
    orig_grid_post = model.grid_expert.post_dit if model.grid_expert is not None else None
    orig_presence = model.grid_expert.post_presence if model.grid_expert is not None and model.grid_expert.presence_head is not None else None

    def grid_post(tokens, pre):
        y = orig_grid_post(tokens, pre)
        captured["grid_pred"] = y
        return y

    def presence(tokens):
        y = orig_presence(tokens)
        captured["presence"] = y
        return y

    if orig_grid_post is not None:
        model.grid_expert.post_dit = grid_post
    if orig_presence is not None:
        model.grid_expert.post_presence = presence
    originals = _patch_timestep_samplers(model, timestep)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        loss, logs = model.training_loss(batch)
        loss.backward()
    finally:
        _restore_timestep_samplers(originals)
        if orig_grid_post is not None:
            model.grid_expert.post_dit = orig_grid_post
        if orig_presence is not None:
            model.grid_expert.post_presence = orig_presence

    grid_params = list(model.grid_expert.named_parameters()) if model.grid_expert is not None else []
    head_params = [(n, p) for n, p in grid_params if "grid_head" in n or "presence_head" in n or "learned_query" in n]
    shared_params = _selected_shared_params(model)
    head_grad, head_names = _flat_grads(head_params)
    shared_grad, shared_names = _flat_grads(shared_params)
    cosine = None
    if head_grad is not None and shared_grad is not None:
        n = min(head_grad.numel(), shared_grad.numel())
        cosine = float(torch.nn.functional.cosine_similarity(head_grad[:n], shared_grad[:n], dim=0).item())

    teacher = batch.get("grid_flow_teacher")
    valid = batch.get("grid_flow_valid")
    moving = None
    angular = 0.0
    binary = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auprc": 0.0}
    if teacher is not None and valid is not None and "grid_pred" in captured:
        teacher_d = teacher.to(device=captured["grid_pred"].device, dtype=captured["grid_pred"].dtype)
        valid_d = valid.to(device=captured["grid_pred"].device, dtype=captured["grid_pred"].dtype)
        moving, _ = model._grid_moving_mask(teacher_d, valid_d)
        pred = captured["grid_pred"].view_as(teacher_d).detach().float()
        target_dir = torch.nn.functional.normalize(teacher_d.float(), dim=-1, eps=1e-6)
        pred_dir = torch.nn.functional.normalize(pred, dim=-1, eps=1e-6)
        mask = moving.bool()
        if mask.any():
            cos = (pred_dir[mask] * target_dir[mask]).sum(dim=-1).clamp(-1, 1)
            angular = float(torch.rad2deg(torch.acos(cos)).mean().item())
        if "presence" in captured:
            logits = captured["presence"].view_as(valid_d)
            binary = _binary_metrics(logits, moving, valid_d)

    return {
        "total_loss": float(loss.detach().float().item()),
        "grid_loss": float(logs.get("loss_grid_raw", 0.0)),
        "direction_loss": float(logs.get("loss_grid_direction", 0.0)),
        "presence_bce": float(logs.get("loss_grid_presence", 0.0)),
        "direction_angular_error_deg": angular,
        **binary,
        "grid_head_grad_norm": _norm([p for _, p in head_params]),
        "shared_backbone_grad_norm": _norm([p for _, p in shared_params]),
        "shared_head_grad_ratio": _norm([p for _, p in shared_params]) / max(_norm([p for _, p in head_params]), 1e-12),
        "primary_grid_gradient_cosine": cosine,
        "selected_head_params": head_names,
        "selected_shared_params": shared_names,
        "logs": {k: float(v) for k, v in logs.items() if isinstance(v, (float, int))},
    }


def _load(config_name: str, checkpoint: str | None, manifest: str | None, precision: str, skip_pretrain_load: bool):
    with initialize_config_dir(config_dir=str(Path.cwd() / "configs"), version_base="1.3"):
        overrides = [f"task={config_name}", f"mixed_precision={precision}", "wandb.enabled=false", "num_workers=0"]
        if checkpoint:
            overrides.append(f"resume={checkpoint}")
        if skip_pretrain_load:
            overrides.append("model.skip_dit_load_from_pretrain=true")
        if manifest:
            overrides.append(f"data.train.manifest_path={manifest}")
        cfg = compose(config_name="train", overrides=overrides)
    if cfg.get("ifwam") is not None and cfg.model.get("ifwam") is None:
        OmegaConf.set_struct(cfg.model, False)
        ifwam_cfg = cfg.ifwam
        if ifwam_cfg.get("enabled") is None and ifwam_cfg.get("ifwam") is not None:
            ifwam_cfg = ifwam_cfg.ifwam
        cfg.model.ifwam = ifwam_cfg
    dtype = _mixed_precision_to_model_dtype(precision)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    if checkpoint:
        model.load_checkpoint(checkpoint)
    model.train()
    ds = instantiate(cfg.data.train)
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0, collate_fn=getattr(ds, "collate_fn", None))
    return cfg, model, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gridfm_one_way_direction_presence_loss_only")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output-dir", default="reports/context_usage_base")
    parser.add_argument("--timesteps", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-pretrain-load", action="store_true")
    args = parser.parse_args()

    cfg, model, loader = _load(args.config, args.checkpoint, args.manifest, args.precision, args.skip_pretrain_load)
    timesteps = [float(x) for x in args.timesteps.split(",") if x.strip()]
    results = {}
    for timestep in timesteps:
        per_variant = {"correct": [], "shuffled": [], "zero": []}
        for bi, batch in enumerate(loader):
            if bi >= args.num_batches:
                break
            for variant in per_variant:
                per_variant[variant].append(_run_variant(model, _context_variant(batch, variant), args.seed, timestep))
        reduced = {}
        for variant, rows in per_variant.items():
            keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and v is not None]
            reduced[variant] = {k: float(sum(float(r[k]) for r in rows) / len(rows)) for k in keys}
            reduced[variant]["selected_head_params"] = rows[0].get("selected_head_params", [])
            reduced[variant]["selected_shared_params"] = rows[0].get("selected_shared_params", [])
        s = reduced["shuffled"]["grid_loss"]
        c = reduced["correct"]["grid_loss"]
        reduced["correct_vs_shuffled_relative_gain"] = None if abs(s) < 1e-12 else float((s - c) / abs(s))
        cosines = [r["primary_grid_gradient_cosine"] for rows in per_variant.values() for r in rows if r["primary_grid_gradient_cosine"] is not None]
        if cosines:
            t = torch.tensor(cosines)
            reduced["gradient_cosine_p10_p50_p90"] = [float(torch.quantile(t, q).item()) for q in (0.1, 0.5, 0.9)]
            reduced["negative_cosine_fraction"] = float((t < 0).float().mean().item())
        results[str(timestep)] = reduced

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    manifest_path = args.manifest or cfg_payload.get("data", {}).get("train", {}).get("manifest_path")
    write_run_artifacts(
        out,
        config=cfg_payload,
        manifest_path=manifest_path,
        checkpoint_path=args.checkpoint,
        extra={"tool": "diagnose_gridflow_context_usage", "timesteps": timesteps, "num_batches": args.num_batches, "skip_pretrain_load": args.skip_pretrain_load},
        repos=[Path.cwd(), "/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow"],
    )
    payload = {"schema": "gridflow_context_usage_v2", "results": results}
    (out / "context_usage.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# GridFM Context Usage", ""]
    for timestep, res in results.items():
        lines.append(f"## timestep {timestep}")
        gain = res.get("correct_vs_shuffled_relative_gain")
        lines.append(f"- correct_vs_shuffled_relative_gain: {gain}")
        for variant in ("correct", "shuffled", "zero"):
            row = res[variant]
            lines.append(f"- {variant}: grid_loss={row['grid_loss']:.6g}, dir={row['direction_loss']:.6g}, pres={row['presence_bce']:.6g}, f1={row['f1']:.6g}, shared/head={row['shared_head_grad_ratio']:.6g}")
        lines.append("")
    (out / "context_usage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
