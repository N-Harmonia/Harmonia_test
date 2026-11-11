from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from georl.utils.seed import RngState


@dataclass(slots=True)
class CheckpointPayload:
    step: int
    epoch: int
    seed: int
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None = None
    scaler: dict[str, Any] | None = None
    rng: RngState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def save_checkpoint_atomic(payload: CheckpointPayload, dest: Path) -> Path:
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    rng = None
    if payload.rng is not None:
        rng = {
            "python": payload.rng.python,
            "numpy": payload.rng.numpy,
            "torch_cpu": payload.rng.torch_cpu,
            "torch_cuda": payload.rng.torch_cuda,
        }
    obj = {
        "step": payload.step,
        "epoch": payload.epoch,
        "seed": payload.seed,
        "model": payload.model,
        "optimizer": payload.optimizer,
        "scheduler": payload.scheduler,
        "scaler": payload.scaler,
        "rng": rng,
        "metadata": payload.metadata,
    }
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=dest.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return dest


def load_checkpoint(path: Path, map_location: torch.device | str = "cpu") -> CheckpointPayload:
    obj = torch.load(path, map_location=map_location, weights_only=False)
    rng = obj.get("rng")
    rng_state: RngState | None = None
    if rng is not None:
        rng_state = RngState(
            python=rng["python"],
            numpy=rng["numpy"],
            torch_cpu=rng["torch_cpu"],
            torch_cuda=rng["torch_cuda"],
        )
    return CheckpointPayload(
        step=int(obj.get("step", 0)),
        epoch=int(obj.get("epoch", 0)),
        seed=int(obj.get("seed", 0)),
        model=obj["model"],
        optimizer=obj["optimizer"],
        scheduler=obj.get("scheduler"),
        scaler=obj.get("scaler"),
        rng=rng_state,
        metadata=obj.get("metadata", {}),
    )
