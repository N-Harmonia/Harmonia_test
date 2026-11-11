from __future__ import annotations

import pytest
import torch

from georl.models.adaptive_tokenizer import AdaptiveTokenizer


@pytest.mark.unit
def test_grid_layout_per_action_fine_medium_coarse_skip() -> None:
    tok = AdaptiveTokenizer(in_channels=3, base_patch=16, embed_dim=8)
    image = torch.randn(1, 3, 32, 32)
    for action_id, expected in ((0, 16 * 4), (1, 4 * 4), (2, 1 * 4), (3, 0)):
        actions = torch.full((1, 4), action_id, dtype=torch.long)
        tokens, mask = tok(image, actions)
        assert tokens.shape[1] == max(expected, 0)
        if expected == 0:
            assert tokens.shape[1] == 0
        else:
            assert mask.sum().item() == expected


@pytest.mark.unit
def test_token_count_helper_matches_paper_budget() -> None:
    a = torch.tensor([[0, 1, 2, 3]])
    assert AdaptiveTokenizer.token_count_per_image(a).item() == 16 + 4 + 1 + 0


@pytest.mark.unit
def test_rejects_non_square_image() -> None:
    tok = AdaptiveTokenizer(in_channels=3, base_patch=16, embed_dim=4)
    image = torch.randn(1, 3, 16, 32)
    with pytest.raises(ValueError):
        tok(image, torch.zeros(1, 2, dtype=torch.long))


@pytest.mark.unit
def test_mixed_actions_concatenate_in_raster_order() -> None:
    tok = AdaptiveTokenizer(in_channels=3, base_patch=16, embed_dim=4)
    image = torch.randn(1, 3, 32, 32)
    actions = torch.tensor([[0, 1, 2, 3]])
    tokens, mask = tok(image, actions)
    assert int(mask.sum().item()) == 16 + 4 + 1
    assert tokens.shape == (1, 21, 4)
