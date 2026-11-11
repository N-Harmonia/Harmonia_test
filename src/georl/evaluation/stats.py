from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, kw_only=True, frozen=True)
class CI:
    mean: float
    lower: float
    upper: float


def bootstrap_ci(
    samples: np.ndarray | list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng_seed: int | None = None,
) -> CI:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty samples")
    rng = np.random.default_rng(rng_seed)
    n = arr.shape[0]
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return CI(mean=float(arr.mean()), lower=float(lo), upper=float(hi))


def multi_seed_summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)
