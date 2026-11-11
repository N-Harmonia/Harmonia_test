from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


@dataclass(slots=True)
class RolloutBuffer:
    capacity: int
    state_dim: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _states: list[torch.Tensor] = field(default_factory=list)
    _actions: list[torch.Tensor] = field(default_factory=list)
    _logp: list[torch.Tensor] = field(default_factory=list)
    _rewards: list[torch.Tensor] = field(default_factory=list)
    _values: list[torch.Tensor] = field(default_factory=list)
    _dones: list[torch.Tensor] = field(default_factory=list)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self._states.append(state.detach().to(self.device))
        self._actions.append(action.detach().to(self.device))
        self._logp.append(log_prob.detach().to(self.device))
        self._rewards.append(reward.detach().to(self.device))
        self._values.append(value.detach().to(self.device))
        self._dones.append(done.detach().to(self.device))

    def __len__(self) -> int:
        return len(self._states)

    @property
    def rewards(self) -> torch.Tensor:
        return torch.stack(self._rewards)

    @property
    def values(self) -> torch.Tensor:
        return torch.stack(self._values)

    @property
    def dones(self) -> torch.Tensor:
        return torch.stack(self._dones)

    def finalize(self, advantages: torch.Tensor, returns: torch.Tensor) -> RolloutBatch:
        return RolloutBatch(
            states=torch.stack(self._states),
            actions=torch.stack(self._actions),
            log_probs=torch.stack(self._logp),
            advantages=advantages,
            returns=returns,
            values=torch.stack(self._values),
        )

    def reset(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._logp.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()


def iter_minibatches(
    batch: RolloutBatch,
    size: int,
    epochs: int,
    generator: torch.Generator | None = None,
) -> Iterator[RolloutBatch]:
    fields = ("states", "actions", "log_probs", "advantages", "returns", "values")
    flat = {k: getattr(batch, k).reshape(-1, *getattr(batch, k).shape[2:]) for k in fields}
    n = next(iter(flat.values())).shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, generator=generator)
        for start in range(0, n, size):
            sl = idx[start : start + size]
            yield RolloutBatch(**{k: v[sl] for k, v in flat.items()})
