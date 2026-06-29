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


def _to_device_sample(sample: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = {kk: vv.clone() if torch.is_tensor(vv) else copy.deepcopy(vv) for kk, vv in v.items()}
        else:
            out[k] = copy.deepcopy(v)
    return out


def _variant(sample: dict[str, Any], name: str) -> dict[str, Any]:
    v = _to_device_sample(sample)
    if name == "correct":
        return v
    if name == "shuffled":
        for key in ("grid_flow_teacher", "grid_flow_valid", "grid_flow_quality"):
            if key in v and torch.is_tensor(v[key]) and v[key].shape[0] > 1:
                v[key] = v[key].flip(0)
        return v
    if name == "zero":
        for key in ("grid_flow_teacher", "grid_flow_valid", "grid_flow_quality"):
            if key in v and torch.is_tensor(v[key]):
                v[key] = torch.zeros_like(v[key])
        if "loss_mask" in v and isinstance(v["loss_mask"], dict) and torch.is_tensor(v["loss_mask"].get("gridflow")):
            v["loss_mask"]["gridflow"] = torch.zeros_like(v["loss_mask"]["gridflow"])
        return v
    if name == "absent":
        for key in ("grid_flow_teacher", "grid_flow_valid", "grid_flow_quality"):
            v.pop(key, None)
        if "loss_mask" in v and isinstance(v["loss_mask"], dict) and torch.is_tensor(v["loss_mask"].get("gridflow")):
            v["loss_mask"]["gridflow"] = torch.zeros_like(v["loss_mask"]["gridflow"])
        return v
    raise ValueError(name)


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    return {"max_abs_diff": float(d.max().item()), "mean_abs_diff": float(d.mean().item())}


def _run_once(model, sample: dict[str, Any], seed: int) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}
    orig_video_post = model.video_expert.post_dit
    orig_action_post = model.action_expert.post_dit

    def video_post(tokens, pre):
        y = orig_video_post(tokens, pre)
        captured["video"] = y.detach()
        return y

    def action_post(tokens, pre):
        y = orig_action_post(tokens, pre)
        captured["action"] = y.detach()
        return y

    model.video_expert.post_dit = video_post
    model.action_expert.post_dit = action_post
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            loss, logs = model.training_loss(sample)
    finally:
        model.video_expert.post_dit = orig_video_post
        model.action_expert.post_dit = orig_action_post
    primary = float(logs.get("loss_video", 0.0)) + float(logs.get("loss_action", 0.0))
    return {
        "loss": float(loss.detach().float().item()),
        "primary_loss": primary,
        "logs": {k: float(v) for k, v in logs.items() if isinstance(v, (float, int))},
        "video": captured["video"],
        "action": captured["action"],
    }


def _make_cfg(config_name: str, interaction_mode: str, checkpoint: str | None, precision: str, skip_pretrain_load: bool):
    config_dir = str(Path.cwd() / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        overrides = [
            f"task={config_name}",
            f"mixed_precision={precision}",
            "wandb.enabled=false",
            "num_workers=0",
        ]
        if checkpoint:
            overrides.append(f"resume={checkpoint}")
        if skip_pretrain_load:
            overrides.append("model.skip_dit_load_from_pretrain=true")
        overrides.append(f"ifwam.ifwam.grid_flow_matching.interaction_mode={interaction_mode}")
        cfg = compose(config_name="train", overrides=overrides)
    if cfg.get("ifwam") is not None and cfg.model.get("ifwam") is None:
        OmegaConf.set_struct(cfg.model, False)
        ifwam_cfg = cfg.ifwam
        if ifwam_cfg.get("enabled") is None and ifwam_cfg.get("ifwam") is not None:
            ifwam_cfg = ifwam_cfg.ifwam
        cfg.model.ifwam = ifwam_cfg
    return cfg


def _run_mode(config_name: str, checkpoint: str | None, precision: str, mode: str, seed: int, skip_pretrain_load: bool):
    cfg = _make_cfg(config_name, mode, checkpoint, precision, skip_pretrain_load)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    if checkpoint:
        model.load_checkpoint(checkpoint)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    ds = instantiate(cfg.data.train)
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0, collate_fn=getattr(ds, "collate_fn", None))
    batch = next(iter(loader))
    variants = {}
    for name in ("correct", "shuffled", "zero", "absent"):
        try:
            variants[name] = _run_once(model, _variant(batch, name), seed)
        except Exception as exc:
            variants[name] = {"error": repr(exc)}

    base = variants["correct"]
    comparisons = {}
    if "error" not in base:
        for name, out in variants.items():
            if name == "correct" or "error" in out:
                continue
            comparisons[name] = {
                "video": _diff(base["video"], out["video"]),
                "action": _diff(base["action"], out["action"]),
                "primary_loss_diff": float(out["primary_loss"] - base["primary_loss"]),
            }
    serial_variants = {}
    for name, out in variants.items():
        serial_variants[name] = {k: v for k, v in out.items() if k not in {"video", "action"}}
    return cfg, {"mode": mode, "variants": serial_variants, "comparisons_vs_correct": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gridfm_one_way_direction_presence_loss_only")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-pretrain-load", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    cfg_for_artifacts = None
    for mode in ("legacy", "one_way_aux"):
        cfg, result = _run_mode(args.config, args.checkpoint, args.precision, mode, args.seed, args.skip_pretrain_load)
        cfg_for_artifacts = OmegaConf.to_container(cfg, resolve=True)
        results[mode] = result

    manifest_path = cfg_for_artifacts.get("data", {}).get("train", {}).get("manifest_path") if isinstance(cfg_for_artifacts, dict) else None
    write_run_artifacts(
        out_dir,
        config=cfg_for_artifacts,
        manifest_path=manifest_path,
        checkpoint_path=args.checkpoint,
        extra={"tool": "production_information_flow_test", "seed": args.seed, "skip_pretrain_load": args.skip_pretrain_load},
        repos=[Path.cwd(), "/2024233240/if-wam_incoming/AutoLabel-3D_Affordance_Flow"],
    )
    payload = {"schema": "production_information_flow_v1", "results": results}
    (out_dir / "production_information_flow_test.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Production Information Flow Test", ""]
    for mode, result in results.items():
        lines.extend([f"## {mode}", ""])
        for variant, comp in result["comparisons_vs_correct"].items():
            lines.append(f"- {variant}: video max={comp['video']['max_abs_diff']:.6g}, action max={comp['action']['max_abs_diff']:.6g}, primary_loss_diff={comp['primary_loss_diff']:.6g}")
        for variant, out in result["variants"].items():
            if "error" in out:
                lines.append(f"- {variant}: ERROR {out['error']}")
        lines.append("")
    (out_dir / "production_information_flow_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
