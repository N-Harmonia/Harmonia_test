"""Task heads on top of the VFM CLS embedding / token sequence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, num_classes)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        cls = encoded[:, 0]
        return self.proj(self.drop(self.norm(cls)))


class SegmentationHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, out_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, num_classes)
        self.out_size = out_size

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tok = encoded[:, 1:]
        logits = self.proj(tok)
        b, n, c = logits.shape
        side = int((n) ** 0.5) or 1
        if side * side != n:
            pad = side * side - n if side * side > n else 0
            logits = F.pad(logits, (0, 0, 0, pad), value=0.0)
            side = int((logits.shape[1]) ** 0.5)
        grid = logits.transpose(1, 2).reshape(b, c, side, side)
        return F.interpolate(grid, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


class DetectionHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, num_queries: int = 100) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.zeros(num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.cls = nn.Linear(embed_dim, num_classes + 1)
        self.bbox = nn.Linear(embed_dim, 4)

    def forward(self, encoded: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = encoded.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        kp = torch.cat([torch.zeros((b, 1), dtype=torch.bool, device=mask.device), ~mask], dim=1)
        attended, _ = self.attn(q, encoded, encoded, key_padding_mask=kp)
        return self.cls(attended), self.bbox(attended).sigmoid()
