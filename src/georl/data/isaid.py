from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

ISAID_CLASSES = (
    "background",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "swimming-pool",
    "roundabout",
    "soccer-ball-field",
    "plane",
    "harbor",
)


@dataclass(slots=True)
class IsaidDataset(Dataset[dict[str, Any]]):
    root: Path
    split: str = "train"
    image_size: int = 512
    ignore_index: int = 255
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    samples: list[tuple[Path, Path]] = field(default_factory=list)

    def __post_init__(self) -> None:
        img_dir = self.root / self.split / "images"
        msk_dir = self.root / self.split / "Semantic_masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"iSAID images dir missing: {img_dir}")
        pairs: list[tuple[Path, Path]] = []
        for img in sorted(img_dir.glob("*.png")):
            mask = msk_dir / img.name.replace(".png", "_instance_color_RGB.png")
            if mask.exists():
                pairs.append((img, mask))
        self.samples = pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        img_path, mask_path = self.samples[idx]
        arr = np.asarray(Image.open(img_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("P")).astype(np.int64)
        sample: dict[str, Any] = {
            "image": arr,
            "mask": torch.from_numpy(mask),
            "image_id": idx,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
