from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

DOTA_V1_CLASSES = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
)


def _parse_dota_label(path: Path) -> list[tuple[np.ndarray, int]]:
    out: list[tuple[np.ndarray, int]] = []
    name_to_id = {n: i for i, n in enumerate(DOTA_V1_CLASSES)}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) < 9 or tokens[0].lower().startswith(("imagesource", "gsd")):
            continue
        try:
            coords = np.array(list(map(float, tokens[:8])), dtype=np.float32).reshape(4, 2)
        except ValueError:
            continue
        cls = tokens[8]
        if cls not in name_to_id:
            continue
        out.append((coords, name_to_id[cls]))
    return out


def _quad_to_xyxy(quad: np.ndarray) -> np.ndarray:
    x1 = quad[:, 0].min()
    y1 = quad[:, 1].min()
    x2 = quad[:, 0].max()
    y2 = quad[:, 1].max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


@dataclass(slots=True)
class DotaDataset(Dataset[dict[str, Any]]):
    root: Path
    split: str = "train"
    image_size: int = 512
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    samples: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        img_dir = self.root / "images" / self.split
        if not img_dir.exists():
            img_dir = self.root / self.split / "images"
        self.samples = sorted(p for p in img_dir.glob("*.png")) + sorted(p for p in img_dir.glob("*.jpg"))
        if not self.samples:
            raise FileNotFoundError(f"no DOTA images under {img_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        img_path = self.samples[idx]
        name = img_path.with_suffix(".txt").name
        candidates = (
            self.root / "labelTxt" / name,
            self.root / self.split / "labelTxt" / name,
            img_path.with_suffix(".txt"),
            img_path.parent.parent / "labelTxt" / name,
        )
        label_path = next((p for p in candidates if p.exists()), candidates[0])
        arr = np.asarray(Image.open(img_path).convert("RGB"))
        labels = _parse_dota_label(label_path)
        boxes = (
            np.stack([_quad_to_xyxy(q) for q, _ in labels]) if labels else np.zeros((0, 4), dtype=np.float32)
        )
        cls_ids = np.array([c for _, c in labels], dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        sample: dict[str, Any] = {
            "image": arr,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(cls_ids),
            "image_id": idx,
            "path": str(img_path),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
