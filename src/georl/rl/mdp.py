from __future__ import annotations

import attrs
import torch


def _gamma_check(_inst: MDP, _attr: attrs.Attribute, value: float) -> None:  # type: ignore[type-arg]
    if not 0.0 <= value < 1.0:
        raise ValueError(f"gamma must be in [0, 1); got {value}")


def _horizon_check(_inst: MDP, _attr: attrs.Attribute, value: int) -> None:  # type: ignore[type-arg]
    if value <= 0:
        raise ValueError(f"horizon must be positive; got {value}")


@attrs.frozen(slots=True, kw_only=True)
class MDP:
    state_dim: int = attrs.field(validator=attrs.validators.instance_of(int))
    num_actions: int = attrs.field(validator=attrs.validators.instance_of(int))
    horizon: int = attrs.field(validator=_horizon_check)
    gamma: float = attrs.field(validator=_gamma_check)


@attrs.frozen(slots=True, kw_only=True)
class RolloutSample:
    state: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    done: torch.Tensor
