from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True, frozen=True)
class RngState:
    python: tuple[Any, ...]
    numpy: dict[str, Any]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]


def set_seed(seed: int, deterministic: bool = False) -> None:
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> RngState:
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]
    return RngState(
        python=random.getstate(),
        numpy=np.random.get_state(legacy=False),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=cuda_states,
    )


def restore_rng_state(state: RngState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if torch.cuda.is_available() and state.torch_cuda:
        for d, s in enumerate(state.torch_cuda):
            torch.cuda.set_rng_state(s, device=d)
