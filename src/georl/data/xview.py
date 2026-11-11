from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class XviewDataset(Dataset[dict[str, Any]]):
    root: Path
    split: str = "train"
    annotation_file: str = "xView_train.geojson"
    image_size: int = 512
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    by_image: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    image_paths: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        ann_path = self.root / self.annotation_file
        if not ann_path.exists():
            raise FileNotFoundError(f"xView annotation missing: {ann_path}")
        data = json.loads(ann_path.read_text())
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            img_id = str(props.get("image_id", ""))
            if not img_id:
                continue
            self.by_image.setdefault(img_id, []).append(props)
        img_dir = self.root / self.split
        self.image_paths = sorted(p for p in img_dir.glob("*.tif"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        path = self.image_paths[idx]
        img = np.asarray(Image.open(path).convert("RGB"))
        anns = self.by_image.get(path.name, [])
        boxes_list: list[list[float]] = []
        labels_list: list[int] = []
        for a in anns:
            bb = a.get("bounds_imcoords")
            cls_id = int(a.get("type_id", -1))
            if not bb or cls_id < 0:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bb.split(","))
            except ValueError:
                continue
            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(cls_id)
        boxes = (
            torch.tensor(boxes_list, dtype=torch.float32)
            if boxes_list
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels = (
            torch.tensor(labels_list, dtype=torch.long)
            if labels_list
            else torch.zeros((0,), dtype=torch.long)
        )
        sample: dict[str, Any] = {
            "image": img,
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "path": str(path),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
