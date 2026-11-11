from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class AmpContext:
    enabled: bool
    dtype: torch.dtype = torch.float16
    scaler: torch.cuda.amp.GradScaler | None = None

    def __post_init__(self) -> None:
        if self.enabled and torch.cuda.is_available() and self.scaler is None:
            self.scaler = torch.cuda.amp.GradScaler()

    def autocast(self) -> Any:
        if self.enabled and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
