from __future__ import annotations

import pytest
import torch

from georl.losses.task import bbox_iou_loss, dice_loss, masked_cross_entropy


@pytest.mark.unit
def test_cross_entropy_ignore_index() -> None:
    logits = torch.randn(8, 5)
    target = torch.tensor([-100, 0, 1, 2, 3, 4, 0, 1])
    loss = masked_cross_entropy(logits, target)
    assert loss.isfinite().item()


@pytest.mark.unit
def test_dice_loss_perfect_prediction_is_zero() -> None:
    target = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 0, 0, 0] = logits[0, 0, 1, 1] = 10.0
    logits[0, 1, 0, 1] = logits[0, 1, 1, 0] = 10.0
    loss = dice_loss(logits, target, num_classes=2)
    assert loss.item() < 0.1


@pytest.mark.unit
def test_bbox_iou_loss_zero_for_identical_boxes() -> None:
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    loss = bbox_iou_loss(boxes, boxes)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
