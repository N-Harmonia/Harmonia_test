"""3-level semantic hierarchy for HSA interpretability."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SemanticHierarchy:
    leaves: tuple[str, ...]
    mid_of: dict[str, str]
    root_of: dict[str, str]

    @property
    def mids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.mid_of.values())))

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.root_of.values())))


_DOTA_TAXONOMY: dict[str, tuple[str, str]] = {
    "plane": ("aircraft", "transport"),
    "helicopter": ("aircraft", "transport"),
    "ship": ("watercraft", "transport"),
    "small-vehicle": ("ground-vehicle", "transport"),
    "large-vehicle": ("ground-vehicle", "transport"),
    "bridge": ("infrastructure", "built-environment"),
    "roundabout": ("infrastructure", "built-environment"),
    "harbor": ("infrastructure", "built-environment"),
    "storage-tank": ("infrastructure", "built-environment"),
    "tennis-court": ("sport-facility", "built-environment"),
    "basketball-court": ("sport-facility", "built-environment"),
    "baseball-diamond": ("sport-facility", "built-environment"),
    "ground-track-field": ("sport-facility", "built-environment"),
    "soccer-ball-field": ("sport-facility", "built-environment"),
    "swimming-pool": ("sport-facility", "built-environment"),
}


def default_hierarchy() -> SemanticHierarchy:
    leaves = tuple(_DOTA_TAXONOMY.keys())
    mid_of = {leaf: mid for leaf, (mid, _) in _DOTA_TAXONOMY.items()}
    root_of = {leaf: root for leaf, (_, root) in _DOTA_TAXONOMY.items()}
    return SemanticHierarchy(leaves=leaves, mid_of=mid_of, root_of=root_of)
