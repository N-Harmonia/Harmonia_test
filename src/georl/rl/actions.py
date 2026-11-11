from __future__ import annotations

from enum import IntEnum
from typing import Final


class Action(IntEnum):
    FINE = 0
    MEDIUM = 1
    COARSE = 2
    SKIP = 3


TOKEN_COUNTS: Final[tuple[int, int, int, int]] = (16, 4, 1, 0)
GRID_SIDES: Final[tuple[int, int, int, int]] = (4, 2, 1, 0)
ACTION_NAMES: Final[tuple[str, str, str, str]] = ("fine", "medium", "coarse", "skip")
NUM_ACTIONS: Final[int] = len(TOKEN_COUNTS)


def tokens_for(a: int | Action) -> int:
    idx = int(a)
    if not 0 <= idx < NUM_ACTIONS:
        raise ValueError(f"action index out of range: {idx}")
    return TOKEN_COUNTS[idx]


def grid_side(a: int | Action) -> int:
    idx = int(a)
    if not 0 <= idx < NUM_ACTIONS:
        raise ValueError(f"action index out of range: {idx}")
    return GRID_SIDES[idx]
