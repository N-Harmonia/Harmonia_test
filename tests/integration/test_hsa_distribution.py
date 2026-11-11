from __future__ import annotations

import pytest
import torch

from georl.hsa.anchor import policy_semantic_alignment
from georl.hsa.hierarchy import default_hierarchy
from georl.rl.actions import Action


@pytest.mark.regression
def test_small_vehicle_disproportionately_fine() -> None:
    h = default_hierarchy()
    classes = h.leaves
    n_per_cls = 200
    region_classes_list: list[int] = []
    actions_list: list[int] = []
    sv_idx = classes.index("small-vehicle")
    plane_idx = classes.index("plane")
    bridge_idx = classes.index("bridge")
    for _ in range(n_per_cls):
        region_classes_list.append(sv_idx)
        actions_list.append(int(Action.FINE))
    for _ in range(n_per_cls):
        region_classes_list.append(plane_idx)
        actions_list.append(int(Action.MEDIUM))
    for _ in range(n_per_cls):
        region_classes_list.append(bridge_idx)
        actions_list.append(int(Action.COARSE))
    region_classes = torch.tensor(region_classes_list)
    actions = torch.tensor(actions_list)
    table = policy_semantic_alignment(region_classes, actions, classes, h)
    sv_dist = table.leaf_to_action["small-vehicle"]
    plane_dist = table.leaf_to_action["plane"]
    bridge_dist = table.leaf_to_action["bridge"]
    assert sv_dist[int(Action.FINE)] > 0.95
    assert plane_dist[int(Action.MEDIUM)] > 0.95
    assert bridge_dist[int(Action.COARSE)] > 0.95
