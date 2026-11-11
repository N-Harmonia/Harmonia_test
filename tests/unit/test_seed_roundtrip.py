from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from georl.utils.seed import capture_rng_state, restore_rng_state, set_seed


@pytest.mark.unit
def test_set_seed_makes_torch_deterministic() -> None:
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)


@pytest.mark.unit
def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)


@pytest.mark.unit
def test_rng_state_round_trip_restores_all_streams() -> None:
    set_seed(7)
    state = capture_rng_state()
    py_a = random.random()
    np_a = float(np.random.rand())
    torch_a = float(torch.randn(()))
    restore_rng_state(state)
    py_b = random.random()
    np_b = float(np.random.rand())
    torch_b = float(torch.randn(()))
    assert py_a == py_b
    assert np_a == np_b
    assert torch_a == torch_b
