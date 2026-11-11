from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return arr


@dataclass(slots=True)
class Sentinel1SarDataset(Dataset[dict[str, Any]]):
    root: Path
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    files: list[Path] = field(default_factory=list)
    expected_bands: int = 2

    def __post_init__(self) -> None:
        self.files = sorted(self.root.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"no SAR tiles under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        arr = _read_npy(self.files[idx])
        if arr.shape[0] != self.expected_bands and arr.shape[-1] == self.expected_bands:
            arr = np.transpose(arr, (2, 0, 1))
        sample: dict[str, Any] = {"image": torch.from_numpy(arr), "modality": "sar", "image_id": idx}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


@dataclass(slots=True)
class Sentinel2MsDataset(Dataset[dict[str, Any]]):
    root: Path
    bands: int = 10
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    files: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.files = sorted(self.root.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"no MS tiles under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        arr = _read_npy(self.files[idx])
        if arr.shape[0] != self.bands and arr.shape[-1] == self.bands:
            arr = np.transpose(arr, (2, 0, 1))
        sample: dict[str, Any] = {"image": torch.from_numpy(arr), "modality": "ms", "image_id": idx}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


@dataclass(slots=True)
class AvirisHsDataset(Dataset[dict[str, Any]]):
    root: Path
    bands: int = 224
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    files: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.files = sorted(self.root.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"no HS tiles under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        arr = _read_npy(self.files[idx])
        if arr.shape[0] != self.bands and arr.shape[-1] == self.bands:
            arr = np.transpose(arr, (2, 0, 1))
        sample: dict[str, Any] = {"image": torch.from_numpy(arr), "modality": "hs", "image_id": idx}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
