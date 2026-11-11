from __future__ import annotations

import pytest
import torch

from georl.models.transformer_block import MultiHeadSelfAttention, TransformerBlock


@pytest.mark.shape
def test_attention_preserves_shape() -> None:
    layer = MultiHeadSelfAttention(dim=32, heads=4)
    x = torch.randn(2, 9, 32)
    assert layer(x).shape == x.shape


@pytest.mark.unit
def test_attention_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(dim=33, heads=4)


@pytest.mark.unit
def test_block_residual_path() -> None:
    blk = TransformerBlock(dim=16, heads=2, ffn_dim=32, dropout=0.0).eval()
    x = torch.randn(1, 8, 16)
    y = blk(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)
