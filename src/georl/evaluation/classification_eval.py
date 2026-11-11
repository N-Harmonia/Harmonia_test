from __future__ import annotations

from collections.abc import Iterable

import torch

from georl.metrics.classification import overall_accuracy


def evaluate_classification(
    predictions: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
) -> dict[str, float]:
    pred = torch.cat([p.flatten() for p in predictions])
    tgt = torch.cat([t.flatten() for t in targets])
    return {"OA": overall_accuracy(pred, tgt)}
