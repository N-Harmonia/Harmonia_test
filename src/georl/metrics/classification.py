from __future__ import annotations

import torch


def overall_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    correct = (pred == target).sum().item()
    return float(correct) / float(target.numel())
