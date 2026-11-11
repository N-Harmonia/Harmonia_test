"""Consistency reward (loss form); Sec. 3.4, Eq. (11)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from georl.rl.actions import NUM_ACTIONS, Action


def raster_neighbour_indices(side: int) -> torch.Tensor:
    n = side * side
    nb = torch.full((n, 8), -1, dtype=torch.long)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for k, (dr, dc) in enumerate(offsets):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    nb[i, k] = rr * side + cc
    return nb


def median_fallback_action(actions_prev: torch.Tensor, seen_mask: torch.Tensor) -> int:
    visible = actions_prev[seen_mask]
    if visible.numel() == 0:
        return int(Action.MEDIUM)
    return int(visible.float().median().item())


def consistency_loss(
    actions_t: torch.Tensor,
    actions_prev: torch.Tensor,
    features: torch.Tensor,
    side: int,
    seen_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative consistency reward, mean over regions."""
    if actions_t.shape != actions_prev.shape:
        raise ValueError("action tensors must share shape")
    n = side * side
    if features.shape[0] != n:
        raise ValueError(f"features must have {n} rows; got {features.shape[0]}")
    nb = raster_neighbour_indices(side).to(features.device)
    feats = F.normalize(features.flatten(1), dim=-1)
    sim_all = feats @ feats.t()
    if seen_mask is None:
        prev = actions_prev
    else:
        fallback = median_fallback_action(actions_prev, seen_mask)
        prev = actions_prev.clone()
        prev[~seen_mask] = fallback
    contrib = sim_all.new_zeros(n)
    for k in range(nb.shape[1]):
        idx = nb[:, k]
        valid = idx >= 0
        if not valid.any():
            continue
        gi = idx.clamp(min=0)
        prev_idx = prev.index_select(0, gi)
        eq = (actions_t == prev_idx) & valid
        sim = sim_all.gather(1, gi.unsqueeze(1)).squeeze(1)
        contrib = contrib + eq.float() * sim
    r = contrib.mean()
    return -r


def class_balance_penalty(actions: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(actions.flatten().long(), minlength=NUM_ACTIONS).float()
    p = counts / counts.sum().clamp(min=1.0)
    target = torch.full_like(p, 1.0 / NUM_ACTIONS)
    return F.kl_div((p + 1e-8).log(), target, reduction="sum")
