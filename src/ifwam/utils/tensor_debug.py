from __future__ import annotations

from typing import Any

import torch


def tensor_summary(x: Any) -> str:
    if not torch.is_tensor(x):
        return f"{type(x).__name__}: {x!r}"
    finite = torch.isfinite(x.detach()).all().item() if x.numel() else True
    return (
        f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
        f"requires_grad={x.requires_grad}, finite={bool(finite)})"
    )
