from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class SceneClassificationDataset(Dataset[dict[str, Any]]):
    root: Path
    image_size: int = 512
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    classes: list[str] = field(default_factory=list)
    samples: list[tuple[Path, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"scene root missing: {self.root}")
        classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        cls_to_id = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for img in sorted((self.root / c).glob("*.jpg")) + sorted((self.root / c).glob("*.png")):
                self.samples.append((img, cls_to_id[c]))
        self.classes = classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        path, label = self.samples[idx]
        img = np.asarray(Image.open(path).convert("RGB"))
        sample: dict[str, Any] = {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": idx,
            "path": str(path),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
