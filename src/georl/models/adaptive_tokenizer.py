"""Adaptive tokenizer; assembles per-region patch tokens given actions."""

from __future__ import annotations

import torch
from torch import nn

from georl.rl.actions import GRID_SIDES, NUM_ACTIONS


class AdaptiveTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_patch: int = 16,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        if base_patch % 4 != 0:
            raise ValueError(f"base_patch must be divisible by 4; got {base_patch}")
        self.base_patch = base_patch
        self.embed_dim = embed_dim
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(in_channels, embed_dim, kernel_size=base_patch // 4, stride=base_patch // 4),
                nn.Conv2d(in_channels, embed_dim, kernel_size=base_patch // 2, stride=base_patch // 2),
                nn.Conv2d(in_channels, embed_dim, kernel_size=base_patch, stride=base_patch),
            ]
        )

    def forward(
        self,
        image: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"image must be 4D, got shape {tuple(image.shape)}")
        b, _, h, w = image.shape
        if h != w:
            raise ValueError("expects square image")
        side = h // self.base_patch
        if actions.shape != (b, side * side):
            raise ValueError(
                f"actions shape {tuple(actions.shape)} != expected (B,N)=({b},{side * side})"
            )

        seqs: list[list[torch.Tensor]] = [[] for _ in range(b)]
        for bi in range(b):
            for ri, a in enumerate(actions[bi].tolist()):
                if not 0 <= a < NUM_ACTIONS:
                    raise ValueError(f"invalid action {a}")
                gs = GRID_SIDES[a]
                if gs == 0:
                    continue
                r, c = divmod(ri, side)
                p = self.base_patch
                patch = image[bi : bi + 1, :, r * p : (r + 1) * p, c * p : (c + 1) * p]
                proj = self.projections[a](patch)
                seqs[bi].append(proj.flatten(2).transpose(1, 2).squeeze(0))

        lengths = [sum(t.shape[0] for t in seq) if seq else 0 for seq in seqs]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            tokens = image.new_zeros((b, 0, self.embed_dim))
            mask = torch.zeros((b, 0), dtype=torch.bool, device=image.device)
            return tokens, mask

        tokens = image.new_zeros((b, max_len, self.embed_dim))
        mask = torch.zeros((b, max_len), dtype=torch.bool, device=image.device)
        for bi, seq in enumerate(seqs):
            if not seq:
                continue
            t = torch.cat(seq, dim=0)
            n = t.shape[0]
            tokens[bi, :n] = t
            mask[bi, :n] = True
        return tokens, mask

    @staticmethod
    def uniform_actions(num_regions: int, action: int, device: torch.device) -> torch.Tensor:
        return torch.full((1, num_regions), action, dtype=torch.long, device=device)

    @staticmethod
    def token_count_per_image(actions: torch.Tensor) -> torch.Tensor:
        counts = torch.tensor([16, 4, 1, 0], device=actions.device)
        return counts.index_select(0, actions.flatten().long()).view_as(actions).sum(dim=-1)
