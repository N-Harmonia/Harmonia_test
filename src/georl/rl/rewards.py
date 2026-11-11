from __future__ import annotations

import attrs
import torch
import torch.nn.functional as F

from georl.rl.actions import TOKEN_COUNTS, Action


@attrs.frozen(slots=True, kw_only=True)
class RewardWeights:
    alpha: float = 1.0
    beta: float = 0.3
    lam: float = 0.1
    n_max: int = max(TOKEN_COUNTS)


def task_reward(loss_prev: torch.Tensor, loss_cur: torch.Tensor) -> torch.Tensor:
    return loss_prev - loss_cur


def cost_reward(actions: torch.Tensor, n_max: int) -> torch.Tensor:
    counts = torch.as_tensor(TOKEN_COUNTS, dtype=torch.float32, device=actions.device)
    n = counts.index_select(0, actions.flatten().long()).view_as(actions)
    return n / float(n_max)


def consistency_reward(
    actions_t: torch.Tensor,
    actions_prev: torch.Tensor,
    features: torch.Tensor,
    neighbour_index: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if actions_t.shape != actions_prev.shape:
        raise ValueError("action tensors must share shape")
    n_regions = features.shape[0]
    feats = F.normalize(features.flatten(1), dim=-1)
    sim = feats @ feats.t()
    same = (actions_t.view(-1, 1) == actions_prev.index_select(0, neighbour_index.flatten()).view(
        n_regions, -1
    )).float()
    pair_sim = sim.gather(1, neighbour_index)
    contrib = same * pair_sim
    if valid_mask is not None:
        contrib = contrib * valid_mask.float()
    return contrib.sum(dim=-1)


def composite_reward(
    r_task: torch.Tensor,
    r_cost: torch.Tensor,
    r_consistency: torch.Tensor,
    weights: RewardWeights,
) -> torch.Tensor:
    return weights.alpha * r_task - weights.beta * r_cost + weights.lam * r_consistency


def median_action_fallback(actions_prev: torch.Tensor, seen_mask: torch.Tensor) -> torch.Tensor:
    seen = actions_prev[seen_mask]
    if seen.numel() == 0:
        median = torch.full_like(actions_prev[:1], int(Action.MEDIUM))
    else:
        median = seen.float().median().long().expand_as(actions_prev[:1])
    out = actions_prev.clone()
    out[~seen_mask] = median.item()
    return out
