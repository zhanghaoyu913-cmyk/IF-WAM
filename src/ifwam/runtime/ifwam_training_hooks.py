from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def infer_action_with_flow_scoring(
    model,
    obs,
    instruction,
    proprio=None,
    num_candidates: int = 4,
    score_threshold: float = 0.5,
    refine_if_low_score: bool = True,
    refine_sigma: float = 0.3,
    **infer_kwargs: Any,
) -> dict[str, Any]:
    """Candidate-scoring inference interface.

    This intentionally does not replace the model's default infer_action path.
    The first implementation calls the default action inference repeatedly and
    uses the optional IF-WAM scoring path when a model exposes it.
    """
    if not hasattr(model, "infer_action"):
        raise AttributeError("model must expose infer_action for IF-WAM candidate scoring")
    actions = []
    scores = []
    infos = []
    for _ in range(int(num_candidates)):
        action = model.infer_action(obs=obs, instruction=instruction, proprio=proprio, **infer_kwargs)
        actions.append(action)
        score = torch.tensor(0.0)
        info = {}
        if hasattr(model, "score_action_candidate"):
            info = model.score_action_candidate(obs=obs, instruction=instruction, action=action, proprio=proprio)
            score = torch.as_tensor(info.get("score", 0.0))
        scores.append(score)
        infos.append(info)
    score_tensor = torch.stack([s.reshape(()) for s in scores])
    best_idx = int(score_tensor.argmax().item())
    used_refinement = False
    best_action = actions[best_idx]
    if refine_if_low_score and float(score_tensor[best_idx].item()) < float(score_threshold) and hasattr(model, "refine_action"):
        best_action = model.refine_action(best_action, sigma=refine_sigma, obs=obs, instruction=instruction, proprio=proprio)
        used_refinement = True
    best_info = infos[best_idx] if infos else {}
    return {
        "action": best_action,
        "score": score_tensor[best_idx],
        "candidate_scores": score_tensor,
        "used_refinement": used_refinement,
        "pred_vflow": best_info.get("pred_vflow"),
        "pred_aflow": best_info.get("pred_aflow"),
    }
