from __future__ import annotations

import pytest
import torch

from georl.models.policy_net import PolicyNetwork
from georl.models.value_net import ValueNetwork
from georl.rl.policy import LinearPolicyHead


@pytest.mark.unit
def test_linear_policy_head_softmax_sums_to_one() -> None:
    head = LinearPolicyHead(state_dim=8, num_actions=4)
    out = head(torch.randn(3, 8))
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)


@pytest.mark.shape
def test_policy_network_logits_shape() -> None:
    net = PolicyNetwork(state_dim=16, num_actions=4, num_blocks=2, heads=2, ffn_dim=32).eval()
    state = torch.randn(2, 16, 16)
    logits = net(state)
    assert logits.shape == (2, 16, 4)
    dist = net.distribution(state)
    assert dist.logits.shape == logits.shape


@pytest.mark.shape
def test_value_network_scalar_per_token() -> None:
    net = ValueNetwork(state_dim=16, num_blocks=2, heads=2, ffn_dim=32, hidden=8).eval()
    state = torch.randn(2, 16, 16)
    v = net(state)
    assert v.shape == (2, 16)
