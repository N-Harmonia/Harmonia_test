from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from georl.data.dota import DotaDataset
from georl.data.loveda import LovedaDataset
from georl.data.scene import SceneClassificationDataset


def _write_rgb(path: Path, size: int = 64) -> None:
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _write_mask(path: Path, size: int = 64, num_classes: int = 7) -> None:
    arr = np.random.randint(0, num_classes, (size, size), dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.mark.integration
def test_dota_reads_quad_labels(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labelTxt"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    _write_rgb(img_dir / "0001.png")
    (lbl_dir / "0001.txt").write_text(
        "imagesource:foo\n"
        "gsd:0.5\n"
        "10 10 30 10 30 30 10 30 small-vehicle 0\n"
        "5 5 25 5 25 25 5 25 large-vehicle 0\n"
    )
    ds = DotaDataset(root=tmp_path, split="train")
    item = ds[0]
    assert item["boxes"].shape[0] == 2
    assert int(item["labels"].max()) >= 0


@pytest.mark.integration
def test_loveda_reads_paired_image_mask(tmp_path: Path) -> None:
    train = tmp_path / "Train" / "Urban"
    (train / "images_png").mkdir(parents=True)
    (train / "masks_png").mkdir(parents=True)
    _write_rgb(train / "images_png" / "0001.png", size=32)
    _write_mask(train / "masks_png" / "0001.png", size=32)
    ds = LovedaDataset(root=tmp_path, split="Train", scene="urban")
    item = ds[0]
    assert item["mask"].shape == (32, 32)


@pytest.mark.integration
def test_scene_walks_class_subdirs(tmp_path: Path) -> None:
    for cls in ("forest", "harbor", "industrial"):
        (tmp_path / cls).mkdir(parents=True)
        for k in range(2):
            _write_rgb(tmp_path / cls / f"{k}.jpg")
    ds = SceneClassificationDataset(root=tmp_path)
    assert len(ds) == 6
    assert sorted(ds.classes) == ["forest", "harbor", "industrial"]
