from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class Resize:
    size: int

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        img = sample["image"]
        if isinstance(img, np.ndarray):
            from cv2 import INTER_LINEAR, resize

            sample["image"] = resize(img, (self.size, self.size), interpolation=INTER_LINEAR)
        elif isinstance(img, torch.Tensor):
            sample["image"] = torch.nn.functional.interpolate(
                img.unsqueeze(0).float(), size=(self.size, self.size), mode="bilinear", align_corners=False
            ).squeeze(0)
        return sample


@dataclass(slots=True)
class RandomFlip:
    p: float = 0.5

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(()) >= self.p:
            return sample
        img = sample["image"]
        if isinstance(img, np.ndarray):
            sample["image"] = img[:, ::-1].copy()
        else:
            sample["image"] = torch.flip(img, dims=(-1,))
        return sample


@dataclass(slots=True)
class RandomRotate90:
    p: float = 0.5

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(()) >= self.p:
            return sample
        k = int(torch.randint(low=1, high=4, size=()).item())
        img = sample["image"]
        if isinstance(img, np.ndarray):
            sample["image"] = np.rot90(img, k=k).copy()
        else:
            sample["image"] = torch.rot90(img, k=k, dims=(-2, -1))
        return sample


@dataclass(slots=True)
class NormalizeBand:
    mean: Sequence[float]
    std: Sequence[float]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        img = sample["image"]
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        mean = torch.tensor(self.mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        sample["image"] = (img - mean) / std
        return sample


@dataclass(slots=True)
class ToTensorChw:
    scale: float = 1.0 / 255.0

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        img = sample["image"]
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]
            t = torch.from_numpy(np.ascontiguousarray(img)).float() * self.scale
            t = t.permute(2, 0, 1)
        else:
            t = img.float()
            if t.ndim == 3 and t.shape[0] not in (1, 3, 4, 10, 13, 224):
                t = t.permute(2, 0, 1) * self.scale
        sample["image"] = t
        return sample


@dataclass(slots=True)
class AugmentationPipeline:
    steps: list[Callable[[dict[str, Any]], dict[str, Any]]]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for step in self.steps:
            sample = step(sample)
        return sample
