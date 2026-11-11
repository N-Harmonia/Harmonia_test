from __future__ import annotations

import math

import torch

from georl.configs.schemas import SchedulerConfig


def cosine_lr(step: int, cfg: SchedulerConfig, base_lr: float) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return base_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    floor = base_lr * cfg.min_lr_factor
    return floor + 0.5 * (base_lr - floor) * (1.0 + math.cos(math.pi * progress))


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: SchedulerConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        return cosine_lr(step, cfg, base_lr=1.0)

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    for pg, lr in zip(optimizer.param_groups, base_lrs, strict=True):
        pg["lr"] = lr
    return sched
