"""HSA explanation template; Sec. 3.7, Eq. (20)."""

from __future__ import annotations

from georl.hsa.anchor import AnchorTable
from georl.rl.actions import ACTION_NAMES

_TEMPLATE = (
    "Action '{action}' is most strongly aligned with the '{category}' category "
    "(level={level}, p={prob:.3f})."
)


def explain_action(action: int, anchor: AnchorTable) -> str:
    if not 0 <= action < len(ACTION_NAMES):
        raise ValueError(f"action out of range: {action}")
    candidates: list[tuple[str, str, float]] = []
    for level, table in (
        ("leaf", anchor.leaf_to_action),
        ("mid", anchor.mid_to_action),
        ("root", anchor.root_to_action),
    ):
        for cat, dist in table.items():
            candidates.append((level, cat, dist.get(action, 0.0)))
    if not candidates:
        return f"Action '{ACTION_NAMES[action]}' has no anchored category."
    best = max(candidates, key=lambda x: x[2])
    return _TEMPLATE.format(
        action=ACTION_NAMES[action],
        category=best[1],
        level=best[0],
        prob=best[2],
    )
