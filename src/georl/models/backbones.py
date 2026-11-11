"""Frozen vision foundation-model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class FrozenVfm(Protocol):
    embed_dim: int

    def encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...


@dataclass(slots=True, kw_only=True)
class VfmSpec:
    name: str
    embed_dim: int
    depth: int
    heads: int
    mlp_ratio: float = 4.0


class VfmAdapter(nn.Module):
    """Thin transformer encoder that consumes already-tokenized inputs."""

    def __init__(self, spec: VfmSpec, dropout: float = 0.0) -> None:
        super().__init__()
        self.spec = spec
        self.embed_dim = spec.embed_dim
        layer = nn.TransformerEncoderLayer(
            d_model=spec.embed_dim,
            nhead=spec.heads,
            dim_feedforward=int(spec.embed_dim * spec.mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.depth)
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.norm = nn.LayerNorm(spec.embed_dim)

    def encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must be (B, N, D)")
        b = tokens.shape[0]
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        key_padding = torch.cat(
            [torch.zeros((b, 1), dtype=torch.bool, device=mask.device), ~mask],
            dim=1,
        )
        out = self.encoder(seq, src_key_padding_mask=key_padding)
        return self.norm(out)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


VFM_SPECS: dict[str, VfmSpec] = {
    "vit_l14": VfmSpec(name="vit_l14", embed_dim=1024, depth=24, heads=16),
    "dinov2_vitl14": VfmSpec(name="dinov2_vitl14", embed_dim=1024, depth=24, heads=16),
    "sam_vit_b": VfmSpec(name="sam_vit_b", embed_dim=768, depth=12, heads=12),
    "remoteclip_vit_l14": VfmSpec(name="remoteclip_vit_l14", embed_dim=1024, depth=24, heads=16),
    "satmae_vit_l16": VfmSpec(name="satmae_vit_l16", embed_dim=1024, depth=24, heads=16),
}


def build_vfm(name: str, dropout: float = 0.0) -> VfmAdapter:
    if name not in VFM_SPECS:
        raise KeyError(f"unknown VFM '{name}'. choices: {sorted(VFM_SPECS)}")
    return VfmAdapter(VFM_SPECS[name], dropout=dropout)
