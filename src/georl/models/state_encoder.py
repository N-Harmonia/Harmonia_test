"""State encoder; Sec. 3.2, Eq. (1)-(5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class LocalBranch(nn.Module):
    """Eq. (2): per-region 3x3 conv tower → 64/128/256."""

    def __init__(self, in_channels: int = 3, out_channels: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor, grid: int) -> torch.Tensor:
        feat = self.stem(image)
        feat = F.adaptive_avg_pool2d(feat, output_size=(grid, grid))
        return rearrange(feat, "b c h w -> b (h w) c")


class GlobalBranch(nn.Module):
    """Eq. (3): 1x1 conv → GAP → single-head attention."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        z = self.proj(image)
        g = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        qkv = self.qkv(g).chunk(3, dim=-1)
        q, k, v = (t.unsqueeze(1) for t in qkv)
        scale = q.shape[-1] ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        ctx = (attn @ v).squeeze(1)
        return self.out(ctx)


class ProgressBranch(nn.Module):
    """Eq. (4): 2-layer MLP over [mask, t/T]."""

    def __init__(self, num_regions: int, hidden: int = 128, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_regions + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, mask_prev: torch.Tensor, t_over_T: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mask_prev.float(), t_over_T.view(-1, 1).float()], dim=-1)
        return self.net(x)


class StateEncoder(nn.Module):
    """Eq. (5): LayerNorm(Linear([f_local; f_global; f_progress]))."""

    def __init__(
        self,
        in_channels: int,
        num_regions: int,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_regions = num_regions
        self.local = LocalBranch(in_channels=in_channels, out_channels=embed_dim)
        self.global_ = GlobalBranch(in_channels=in_channels, embed_dim=embed_dim)
        self.progress = ProgressBranch(num_regions=num_regions, out_dim=embed_dim)
        self.proj = nn.Linear(3 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def grid_side(self) -> int:
        side = int(self.num_regions**0.5)
        if side * side != self.num_regions:
            raise ValueError(f"num_regions ({self.num_regions}) is not a perfect square")
        return side

    def forward(
        self,
        image: torch.Tensor,
        mask_prev: torch.Tensor,
        t_over_T: torch.Tensor,
    ) -> torch.Tensor:
        side = self.grid_side
        local = self.local(image, grid=side)
        global_ = self.global_(image).unsqueeze(1).expand(-1, side * side, -1)
        progress = self.progress(mask_prev, t_over_T).unsqueeze(1).expand(-1, side * side, -1)
        cat = torch.cat([local, global_, progress], dim=-1)
        return self.norm(self.proj(cat))
