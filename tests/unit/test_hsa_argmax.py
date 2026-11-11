from __future__ import annotations

import pytest
import torch

from georl.hsa.anchor import policy_semantic_alignment
from georl.hsa.explain import explain_action
from georl.hsa.hierarchy import default_hierarchy
from georl.rl.actions import Action


@pytest.mark.unit
def test_anchor_concentrates_on_dominant_action_per_category() -> None:
    h = default_hierarchy()
    classes = h.leaves
    n = 32
    region_classes = torch.zeros(n, dtype=torch.long)
    region_classes[16:] = classes.index("small-vehicle")
    actions = torch.zeros(n, dtype=torch.long)
    actions[16:] = int(Action.FINE)
    actions[:16] = int(Action.COARSE)
    table = policy_semantic_alignment(region_classes, actions, classes, h)
    plane_dist = table.leaf_to_action["plane"]
    sv_dist = table.leaf_to_action["small-vehicle"]
    assert plane_dist[int(Action.COARSE)] == pytest.approx(1.0)
    assert sv_dist[int(Action.FINE)] == pytest.approx(1.0)


@pytest.mark.unit
def test_explain_template_uses_argmax_category() -> None:
    h = default_hierarchy()
    classes = h.leaves
    region_classes = torch.tensor([classes.index("small-vehicle")] * 4)
    actions = torch.full((4,), int(Action.FINE), dtype=torch.long)
    table = policy_semantic_alignment(region_classes, actions, classes, h)
    msg = explain_action(int(Action.FINE), table)
    assert "fine" in msg
    assert "small-vehicle" in msg or "transport" in msg
