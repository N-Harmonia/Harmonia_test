from __future__ import annotations

import pytest
import torch

from georl.models.state_encoder import GlobalBranch, LocalBranch, ProgressBranch, StateEncoder


@pytest.mark.shape
def test_local_branch_output_shape() -> None:
    branch = LocalBranch(in_channels=3, out_channels=32)
    image = torch.randn(2, 3, 64, 64)
    out = branch(image, grid=4)
    assert out.shape == (2, 16, 32)


@pytest.mark.shape
def test_global_branch_output_shape() -> None:
    branch = GlobalBranch(in_channels=3, embed_dim=32)
    out = branch(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 32)


@pytest.mark.shape
def test_progress_branch_output_shape() -> None:
    branch = ProgressBranch(num_regions=16, hidden=16, out_dim=32)
    mask = torch.zeros(2, 16)
    t = torch.tensor([0.5, 1.0])
    assert branch(mask, t).shape == (2, 32)


@pytest.mark.shape
def test_state_encoder_eq5_concat_and_proj() -> None:
    enc = StateEncoder(in_channels=3, num_regions=16, embed_dim=32)
    image = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 16)
    t = torch.tensor([0.25, 1.0])
    out = enc(image, mask, t)
    assert out.shape == (2, 16, 32)
    assert torch.isfinite(out).all()


@pytest.mark.shape
def test_state_encoder_requires_square_grid() -> None:
    with pytest.raises(ValueError):
        StateEncoder(in_channels=3, num_regions=15, embed_dim=8).grid_side
