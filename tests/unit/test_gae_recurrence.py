from __future__ import annotations

import pytest
import torch

from georl.rl.gae import compute_gae, compute_gae_recursive


@pytest.mark.unit
def test_two_paths_agree() -> None:
    torch.manual_seed(7)
    horizon, batch = 6, 3
    rewards = torch.randn(horizon, batch)
    values = torch.randn(horizon, batch)
    dones = torch.zeros(horizon, batch)
    dones[-1] = 1.0
    bootstrap = torch.zeros(batch)
    adv_a, ret_a = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, bootstrap_value=bootstrap)
    adv_b = compute_gae_recursive(
        rewards, values, dones, gamma=0.99, lam=0.95, bootstrap_value=bootstrap
    )
    assert torch.allclose(adv_a, adv_b, atol=1e-5)
    assert torch.allclose(ret_a, adv_a + values)


@pytest.mark.unit
def test_dones_truncate_gae_propagation() -> None:
    horizon, batch = 4, 1
    rewards = torch.ones(horizon, batch)
    values = torch.zeros(horizon, batch)
    dones = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert adv[1].item() == pytest.approx(1.0)
    assert adv[3].item() == pytest.approx(1.0)
    assert ret.shape == rewards.shape
