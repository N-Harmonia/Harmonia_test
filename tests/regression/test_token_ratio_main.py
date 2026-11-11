from __future__ import annotations

import pytest
import torch

from georl.metrics.efficiency import token_ratio


@pytest.mark.regression
def test_token_ratio_for_paper_target_band_37_53() -> None:
    torch.manual_seed(1)
    n_regions = 1024
    actions = torch.zeros(1, n_regions, dtype=torch.long)
    actions[0, :400] = 0
    actions[0, 400:700] = 1
    actions[0, 700:900] = 2
    actions[0, 900:] = 3
    ratio = token_ratio(actions)
    assert 0.37 <= ratio <= 0.53


@pytest.mark.regression
def test_token_ratio_monotonic_when_skips_increase() -> None:
    base = torch.zeros(1, 16, dtype=torch.long)
    r_low = token_ratio(base)
    half_skip = base.clone()
    half_skip[0, ::2] = 3
    r_half = token_ratio(half_skip)
    all_skip = torch.full_like(base, 3)
    r_zero = token_ratio(all_skip)
    assert r_low > r_half > r_zero
