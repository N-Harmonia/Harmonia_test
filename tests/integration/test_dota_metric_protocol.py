from __future__ import annotations

import pytest
import torch

from georl.metrics.detection import mean_average_precision


@pytest.mark.integration
def test_perfect_predictions_yield_unit_map() -> None:
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [50.0, 50.0, 80.0, 80.0]]),
            "labels": torch.tensor([0, 1]),
        }
    ]
    predictions = [
        {
            "boxes": targets[0]["boxes"].clone(),
            "scores": torch.tensor([0.95, 0.92]),
            "labels": targets[0]["labels"].clone(),
        }
    ]
    out = mean_average_precision(predictions, targets, num_classes=2, iou_threshold=0.5)
    assert out["mAP_0.5"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.integration
def test_low_iou_predictions_score_zero() -> None:
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]), "labels": torch.tensor([0])}
    ]
    predictions = [
        {
            "boxes": torch.tensor([[100.0, 100.0, 130.0, 130.0]]),
            "scores": torch.tensor([0.99]),
            "labels": torch.tensor([0]),
        }
    ]
    out = mean_average_precision(predictions, targets, num_classes=1, iou_threshold=0.5)
    assert out["mAP_0.5"] == pytest.approx(0.0, abs=1e-6)
