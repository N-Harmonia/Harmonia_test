from __future__ import annotations

import pytest
import torch

from georl.losses.consistency import (
    median_fallback_action,
    raster_neighbour_indices,
)
from georl.rl.actions import Action


@pytest.mark.unit
def test_raster_neighbours_shape_and_corner_sentinels() -> None:
    nb = raster_neighbour_indices(3)
    assert nb.shape == (9, 8)
    corner = nb[0]
    assert (corner == -1).sum().item() == 5


@pytest.mark.unit
def test_median_fallback_returns_median_of_seen() -> None:
    actions_prev = torch.tensor([0, 1, 2, 3])
    seen = torch.tensor([True, True, False, False])
    fallback = median_fallback_action(actions_prev, seen)
    assert fallback in (0, 1)


@pytest.mark.unit
def test_median_fallback_handles_empty_seen() -> None:
    actions_prev = torch.tensor([0, 1])
    seen = torch.tensor([False, False])
    assert median_fallback_action(actions_prev, seen) == int(Action.MEDIUM)
