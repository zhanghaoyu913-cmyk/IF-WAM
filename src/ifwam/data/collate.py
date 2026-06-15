from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

LOSS_KEYS = ("video", "action", "vflow", "aflow", "gridflow", "mag", "progress")


def _stack_optional(batch: list[dict[str, Any]], key: str) -> torch.Tensor | None:
    values = [item.get(key) for item in batch]
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        template = next(v for v in values if v is not None)
        values = [torch.zeros_like(template) if v is None else v for v in values]
    return torch.stack(values, dim=0)


def collate_ifwam_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty IF-WAM batch.")

    out: dict[str, Any] = {}
    tensor_keys = (
        "video",
        "action",
        "action_mask",
        "state",
        "state_mask",
        "proprio",
        "proprio_mask",
        "iflow_teacher",
        "iflow_valid",
        "iflow_quality",
        "iflow_mask",
        "grid_flow_teacher",
        "grid_flow_valid",
        "grid_flow_quality",
        "grid_flow_mask",
        "image_is_pad",
        "action_is_pad",
        "proprio_is_pad",
        "context",
        "context_mask",
    )
    for key in tensor_keys:
        value = _stack_optional(batch, key)
        if value is not None:
            out[key] = value

    loss_mask = {}
    for key in LOSS_KEYS:
        loss_mask[key] = torch.tensor(
            [float(item.get("loss_mask", {}).get(key, 0.0)) for item in batch],
            dtype=torch.float32,
        )
    out["loss_mask"] = loss_mask

    for key in ("language", "prompt", "dataset_family", "source_dataset", "task_label", "traj_dir", "layout_key"):
        out[key] = [item.get(key) for item in batch]
    out["semantics"] = [item.get("semantics", {}) for item in batch]

    # Fast-WAM expects `prompt`; IF-WAM schema uses `language`.
    if "prompt" not in out or all(v is None for v in out["prompt"]):
        out["prompt"] = out["language"]
    return out
