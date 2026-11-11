from __future__ import annotations

import pytest

from georl.rl.actions import GRID_SIDES, NUM_ACTIONS, TOKEN_COUNTS, Action, grid_side, tokens_for


@pytest.mark.unit
def test_eq6_token_budget() -> None:
    assert TOKEN_COUNTS == (16, 4, 1, 0)
    assert tokens_for(Action.FINE) == 16
    assert tokens_for(Action.MEDIUM) == 4
    assert tokens_for(Action.COARSE) == 1
    assert tokens_for(Action.SKIP) == 0


@pytest.mark.unit
def test_grid_sides_match_token_counts() -> None:
    for a, n in enumerate(TOKEN_COUNTS):
        gs = GRID_SIDES[a]
        assert (gs * gs) == n


@pytest.mark.unit
def test_grid_side_helper_matches_table() -> None:
    for a in range(NUM_ACTIONS):
        assert grid_side(a) == GRID_SIDES[a]


@pytest.mark.unit
def test_invalid_action_raises() -> None:
    with pytest.raises(ValueError):
        tokens_for(99)
    with pytest.raises(ValueError):
        grid_side(-1)
