from __future__ import annotations

import pytest
import torch
from torch import nn

from georl.models.policy_net import PolicyNetwork
from georl.models.state_encoder import StateEncoder


@pytest.mark.overfit
def test_policy_can_overfit_single_class_label() -> None:
    torch.manual_seed(0)
    encoder = StateEncoder(in_channels=3, num_regions=4, embed_dim=16)
    policy = PolicyNetwork(state_dim=16, num_actions=4, num_blocks=1, heads=2, ffn_dim=32)
    image = torch.randn(2, 3, 32, 32)
    mask = torch.zeros(2, 4)
    t = torch.tensor([1.0, 1.0])
    target = torch.zeros(2, 4, dtype=torch.long)
    optim = torch.optim.AdamW(list(encoder.parameters()) + list(policy.parameters()), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    final_loss = None
    for _ in range(200):
        optim.zero_grad(set_to_none=True)
        state = encoder(image, mask, t)
        logits = policy(state)
        loss = ce(logits.reshape(-1, 4), target.reshape(-1))
        loss.backward()
        optim.step()
        final_loss = loss.item()
    assert final_loss is not None
    assert final_loss < 0.05
