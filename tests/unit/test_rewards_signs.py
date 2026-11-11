from __future__ import annotations

import pytest
import torch

from georl.rl.actions import TOKEN_COUNTS, Action
from georl.rl.rewards import RewardWeights, composite_reward, cost_reward, task_reward


@pytest.mark.unit
def test_task_reward_positive_when_loss_decreases() -> None:
    prev = torch.tensor([1.0, 1.0])
    cur = torch.tensor([0.4, 0.7])
    r = task_reward(prev, cur)
    assert torch.all(r > 0)


@pytest.mark.unit
def test_cost_reward_scales_with_token_budget() -> None:
    actions = torch.tensor([[Action.FINE, Action.MEDIUM, Action.COARSE, Action.SKIP]])
    n_max = max(TOKEN_COUNTS)
    r = cost_reward(actions, n_max)
    expected = torch.tensor([[16 / n_max, 4 / n_max, 1 / n_max, 0.0]])
    assert torch.allclose(r, expected)


@pytest.mark.unit
def test_composite_obeys_eq8_signs() -> None:
    w = RewardWeights(alpha=1.0, beta=0.3, lam=0.1, n_max=16)
    r_task = torch.tensor([0.5])
    r_cost = torch.tensor([1.0])
    r_cons = torch.tensor([0.2])
    out = composite_reward(r_task, r_cost, r_cons, w)
    expected = 1.0 * 0.5 - 0.3 * 1.0 + 0.1 * 0.2
    assert torch.allclose(out, torch.tensor([expected]))
