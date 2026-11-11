from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup_ddp() -> torch.device:
    if "RANK" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def teardown_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_wrap_ddp(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not (dist.is_available() and dist.is_initialized()):
        return module
    return DistributedDataParallel(
        module,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
    )
