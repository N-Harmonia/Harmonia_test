from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

LOVEDA_CLASSES = ("background", "building", "road", "water", "barren", "forest", "agricultural")

Scene = Literal["urban", "rural"]


@dataclass(slots=True)
class LovedaDataset(Dataset[dict[str, Any]]):
    root: Path
    split: str = "Train"
    scene: Scene = "urban"
    image_size: int = 512
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    pairs: list[tuple[Path, Path]] = field(default_factory=list)

    def __post_init__(self) -> None:
        scene_dir = self.root / self.split / self.scene.capitalize()
        img_dir = scene_dir / "images_png"
        msk_dir = scene_dir / "masks_png"
        if not img_dir.exists():
            raise FileNotFoundError(f"LoveDA images missing: {img_dir}")
        for img in sorted(img_dir.glob("*.png")):
            mask = msk_dir / img.name
            if mask.exists():
                self.pairs.append((img, mask))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        img_path, mask_path = self.pairs[idx]
        arr = np.asarray(Image.open(img_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path)).astype(np.int64)
        sample: dict[str, Any] = {"image": arr, "mask": torch.from_numpy(mask), "image_id": idx}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
