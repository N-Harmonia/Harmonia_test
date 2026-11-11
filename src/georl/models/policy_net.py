"""Policy network; Sec. 3.5, Eq. (12)."""

from __future__ import annotations

import torch
from torch import nn

from georl.models.transformer_block import TransformerBlock


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 256,
        num_actions: int = 4,
        num_blocks: int = 4,
        heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=state_dim, heads=heads, ffn_dim=ffn_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(state_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

    def distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)
