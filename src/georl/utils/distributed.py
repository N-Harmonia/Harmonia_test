from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def init_from_env() -> torch.device:
    if "RANK" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def destroy() -> None:
    if is_distributed():
        dist.destroy_process_group()
