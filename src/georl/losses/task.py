"""Task losses (Eq. 9 specialisations)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(logits, target, ignore_index=ignore_index, weight=weight)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
    ignore_index: int = -100,
) -> torch.Tensor:
    valid = target != ignore_index
    t_clamped = target.clamp(min=0)
    onehot = F.one_hot(t_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
    onehot = onehot * valid.unsqueeze(1).float()
    probs = logits.softmax(dim=1)
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bbox_iou_loss(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    if pred_xyxy.shape != target_xyxy.shape:
        raise ValueError("bbox tensors must share shape")
    x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=0)
    area_t = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=0) * (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp(min=0)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union
    return (1.0 - iou).mean()
