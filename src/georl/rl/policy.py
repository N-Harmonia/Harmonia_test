from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LinearPolicyHead(nn.Module):
    """Linear softmax policy head; Eq. 7."""

    def __init__(self, state_dim: int, num_actions: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, num_actions, bias=bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.linear(state)
        return F.softmax(logits, dim=-1)

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)
