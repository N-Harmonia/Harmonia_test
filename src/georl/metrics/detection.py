"""DOTA-style mAP@0.5; axis-aligned approximation for the harness."""

from __future__ import annotations

import numpy as np
import torch


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _voc_ap_2007(recall: np.ndarray, precision: np.ndarray) -> float:
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        mask = recall >= r
        p = float(precision[mask].max()) if mask.any() else 0.0
        ap += p / 11.0
    return ap


def _per_class_ap(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_image_ids: np.ndarray,
    gt_boxes: np.ndarray,
    gt_image_ids: np.ndarray,
    iou_threshold: float,
) -> float:
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_image_ids = pred_image_ids[order]
    gt_count = gt_boxes.shape[0]
    if gt_count == 0:
        return 0.0
    matched: dict[int, set[int]] = {int(i): set() for i in np.unique(gt_image_ids).tolist()}
    tp = np.zeros(pred_boxes.shape[0], dtype=np.float32)
    fp = np.zeros(pred_boxes.shape[0], dtype=np.float32)
    for pi, (box, img_id) in enumerate(zip(pred_boxes, pred_image_ids, strict=True)):
        gt_idx = np.where(gt_image_ids == img_id)[0]
        if gt_idx.size == 0:
            fp[pi] = 1.0
            continue
        ious = _box_iou_matrix(box[None, :], gt_boxes[gt_idx])[0]
        best = int(ious.argmax())
        if ious[best] >= iou_threshold and gt_idx[best] not in matched[int(img_id)]:
            tp[pi] = 1.0
            matched[int(img_id)].add(int(gt_idx[best]))
        else:
            fp[pi] = 1.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(gt_count, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    return _voc_ap_2007(recall, precision)


def mean_average_precision(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    per_cls = []
    for cls in range(num_classes):
        pbs, pss, pis = [], [], []
        gbs, gis = [], []
        for img_id, (pred, gt) in enumerate(zip(predictions, targets, strict=True)):
            p_mask = pred["labels"] == cls
            g_mask = gt["labels"] == cls
            pbs.append(pred["boxes"][p_mask].detach().cpu().numpy())
            pss.append(pred["scores"][p_mask].detach().cpu().numpy())
            pis.append(np.full(int(p_mask.sum().item()), img_id, dtype=np.int64))
            gbs.append(gt["boxes"][g_mask].detach().cpu().numpy())
            gis.append(np.full(int(g_mask.sum().item()), img_id, dtype=np.int64))
        ap = _per_class_ap(
            pred_boxes=np.concatenate(pbs) if pbs else np.zeros((0, 4)),
            pred_scores=np.concatenate(pss) if pss else np.zeros((0,)),
            pred_image_ids=np.concatenate(pis) if pis else np.zeros((0,), dtype=np.int64),
            gt_boxes=np.concatenate(gbs) if gbs else np.zeros((0, 4)),
            gt_image_ids=np.concatenate(gis) if gis else np.zeros((0,), dtype=np.int64),
            iou_threshold=iou_threshold,
        )
        per_cls.append(ap)
    mAP = float(np.mean(per_cls)) if per_cls else 0.0
    return {"mAP_0.5": mAP, "per_class": float(np.array(per_cls).mean())}
