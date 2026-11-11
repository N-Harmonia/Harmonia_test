from __future__ import annotations

from collections.abc import Iterable

import torch

from georl.metrics.segmentation import mean_iou, mean_iou_ignore_background


def evaluate_segmentation(
    predictions: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
    num_classes: int,
    ignore_index: int = 255,
    ignore_background: bool = False,
) -> dict[str, float]:
    pred_cat = torch.cat([p.flatten() for p in predictions])
    tgt_cat = torch.cat([t.flatten() for t in targets])
    if ignore_background:
        miou = mean_iou_ignore_background(pred_cat, tgt_cat, num_classes, ignore_index)
    else:
        miou = mean_iou(pred_cat, tgt_cat, num_classes, ignore_index)
    correct = (pred_cat == tgt_cat) & (tgt_cat != ignore_index)
    valid = (tgt_cat != ignore_index).sum().clamp(min=1)
    oa = float(correct.sum().float() / valid.float())
    return {"mIoU": miou, "OA": oa}
