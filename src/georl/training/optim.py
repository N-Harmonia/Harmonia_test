from __future__ import annotations

from collections.abc import Iterable

import torch

from georl.configs.schemas import OptimConfig


def build_optimizer(parameters: Iterable[torch.nn.Parameter], cfg: OptimConfig) -> torch.optim.Optimizer:
    decay, no_decay = _split_decay_parameters(parameters)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.name.lower() == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    if cfg.name.lower() == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    if cfg.name.lower() == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9)
    raise ValueError(f"unknown optimizer: {cfg.name}")


def _split_decay_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for p in parameters:
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay
