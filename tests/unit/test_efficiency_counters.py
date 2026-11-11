from __future__ import annotations

import pytest
import torch

from georl.metrics.efficiency import token_ratio
from georl.rl.actions import TOKEN_COUNTS, Action


@pytest.mark.unit
def test_token_ratio_all_fine_is_one() -> None:
    actions = torch.full((2, 4), int(Action.FINE), dtype=torch.long)
    assert token_ratio(actions) == pytest.approx(1.0)


@pytest.mark.unit
def test_token_ratio_all_skip_is_zero() -> None:
    actions = torch.full((1, 4), int(Action.SKIP), dtype=torch.long)
    assert token_ratio(actions) == pytest.approx(0.0)


@pytest.mark.unit
def test_token_ratio_mixed_actions() -> None:
    actions = torch.tensor([[Action.FINE, Action.MEDIUM, Action.COARSE, Action.SKIP]])
    expected = (16 + 4 + 1 + 0) / (4 * max(TOKEN_COUNTS))
    assert token_ratio(actions) == pytest.approx(expected)
