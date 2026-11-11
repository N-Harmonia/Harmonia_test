from __future__ import annotations

import torch


def _confusion(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    mask = target != ignore_index
    p = pred[mask].long()
    t = target[mask].long()
    k = (p >= 0) & (p < num_classes) & (t >= 0) & (t < num_classes)
    idx = t[k] * num_classes + p[k]
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def per_class_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    cm = _confusion(pred, target, num_classes, ignore_index)
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    denom = (tp + fp + fn).clamp(min=1.0)
    return tp / denom


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    return float(per_class_iou(pred, target, num_classes, ignore_index).mean().item())


def mean_iou_ignore_background(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255
) -> float:
    iou = per_class_iou(pred, target, num_classes, ignore_index)
    return float(iou[1:].mean().item())
