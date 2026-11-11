from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from georl.metrics.detection import mean_average_precision


def evaluate_detection(
    predictions: Iterable[dict[str, torch.Tensor]],
    targets: Iterable[dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    return mean_average_precision(
        predictions=list(predictions),
        targets=list(targets),
        num_classes=num_classes,
        iou_threshold=iou_threshold,
    )
