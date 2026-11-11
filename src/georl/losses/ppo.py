"""PPO loss; Sec. 3.6, Eq. (16) and Eq. (18)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True, kw_only=True)
class PpoOutput:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor


def ppo_clip_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = (new_log_probs - old_log_probs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
    return policy_loss, approx_kl, clip_frac


def value_loss_fn(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_eps: float | None = None,
) -> torch.Tensor:
    if clip_eps is None:
        return F.mse_loss(new_values, returns)
    clipped = old_values + (new_values - old_values).clamp(-clip_eps, clip_eps)
    return torch.max(F.mse_loss(new_values, returns, reduction="none"),
                     F.mse_loss(clipped, returns, reduction="none")).mean()


def total_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    c_value: float = 0.5,
    c_entropy: float = 0.01,
    value_clip_eps: float | None = None,
) -> PpoOutput:
    pi_loss, kl, cf = ppo_clip_loss(new_log_probs, old_log_probs, advantages, clip_eps=clip_eps)
    v_loss = value_loss_fn(new_values, old_values, returns, clip_eps=value_clip_eps)
    h = entropy.mean()
    total = pi_loss + c_value * v_loss - c_entropy * h
    return PpoOutput(
        policy_loss=pi_loss,
        value_loss=v_loss,
        entropy=h,
        total=total,
        approx_kl=kl,
        clip_frac=cf,
    )
