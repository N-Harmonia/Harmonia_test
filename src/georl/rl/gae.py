from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    bootstrap_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, dones must share shape (T, ...)")
    horizon = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    if bootstrap_value is None:
        bootstrap_value = torch.zeros_like(values[0])
    next_value = bootstrap_value
    gae = torch.zeros_like(values[0])
    for t in reversed(range(horizon)):
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def compute_gae_recursive(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    bootstrap_value: torch.Tensor,
) -> torch.Tensor:
    horizon = rewards.shape[0]
    deltas = torch.empty_like(rewards)
    next_v = bootstrap_value
    for t in reversed(range(horizon)):
        not_done = 1.0 - dones[t].float()
        deltas[t] = rewards[t] + gamma * next_v * not_done - values[t]
        next_v = values[t]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(values[0])
    for t in reversed(range(horizon)):
        not_done = 1.0 - dones[t].float()
        running = deltas[t] + gamma * lam * not_done * running
        adv[t] = running
    return adv
