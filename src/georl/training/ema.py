from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(slots=True)
class ExponentialMovingAverage:
    decay: float
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)
    backup: dict[str, torch.Tensor] = field(default_factory=dict)

    def register(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow.get(name)
            if s is None:
                self.shadow[name] = p.detach().clone()
                continue
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    def restore(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()
