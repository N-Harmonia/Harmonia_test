"""Policy-semantic alignment; Sec. 3.7, Eq. (19)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from georl.hsa.hierarchy import SemanticHierarchy
from georl.rl.actions import NUM_ACTIONS


@dataclass(slots=True)
class AnchorTable:
    hierarchy: SemanticHierarchy
    leaf_to_action: dict[str, dict[int, float]] = field(default_factory=dict)
    mid_to_action: dict[str, dict[int, float]] = field(default_factory=dict)
    root_to_action: dict[str, dict[int, float]] = field(default_factory=dict)


def policy_semantic_alignment(
    region_classes: torch.Tensor,
    actions: torch.Tensor,
    class_names: tuple[str, ...],
    hierarchy: SemanticHierarchy,
    weights: torch.Tensor | None = None,
) -> AnchorTable:
    if region_classes.shape != actions.shape:
        raise ValueError("region_classes and actions must share shape")
    n = region_classes.numel()
    if weights is None:
        weights = torch.ones(n, device=region_classes.device)
    weights = weights.flatten()
    cls = region_classes.flatten().long()
    act = actions.flatten().long()
    table = AnchorTable(hierarchy=hierarchy)
    for level, lookup, dest in (
        ("leaf", {i: name for i, name in enumerate(class_names)}, table.leaf_to_action),
        ("mid", {i: hierarchy.mid_of.get(class_names[i], "other") for i in range(len(class_names))}, table.mid_to_action),
        ("root", {i: hierarchy.root_of.get(class_names[i], "other") for i in range(len(class_names))}, table.root_to_action),
    ):
        denom: dict[str, torch.Tensor] = {}
        nums: dict[str, torch.Tensor] = {}
        for i in range(n):
            key = lookup[int(cls[i].item())]
            denom[key] = denom.get(key, torch.zeros((), device=region_classes.device)) + weights[i]
            v = nums.setdefault(key, torch.zeros(NUM_ACTIONS, device=region_classes.device))
            v[int(act[i].item())] = v[int(act[i].item())] + weights[i]
        for key, vec in nums.items():
            p = (vec / denom[key].clamp(min=1e-9)).tolist()
            dest[key] = {a: float(p[a]) for a in range(NUM_ACTIONS)}
        _ = level
    return table
