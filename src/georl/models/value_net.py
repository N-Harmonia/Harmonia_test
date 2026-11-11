"""Value network; Sec. 3.5, Eq. (15)."""

from __future__ import annotations

import torch
from torch import nn

from georl.models.transformer_block import TransformerBlock


class ValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 256,
        num_blocks: int = 4,
        heads: int = 4,
        ffn_dim: int = 1024,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=state_dim, heads=heads, ffn_dim=ffn_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(-1)
