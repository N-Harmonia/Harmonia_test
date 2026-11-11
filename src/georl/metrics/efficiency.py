"""Computational efficiency counters; Table 6."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from georl.rl.actions import TOKEN_COUNTS


@dataclass(slots=True, kw_only=True)
class EfficiencyReport:
    gflops: float
    latency_ms: float
    peak_memory_mb: float
    token_ratio: float


def estimate_gflops(model: nn.Module, sample: torch.Tensor) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError("fvcore not installed; install fvcore to estimate GFLOPs") from exc
    flops = FlopCountAnalysis(model, sample).unsupported_ops_warnings(False).total()
    return float(flops) / 1e9


def measure_latency(model: nn.Module, sample: torch.Tensor, repeats: int = 50, warmup: int = 5) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if sample.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if sample.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / repeats


def measure_peak_memory(model: nn.Module, sample: torch.Tensor) -> float:
    if sample.device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(sample.device)
    with torch.no_grad():
        model(sample)
    return torch.cuda.max_memory_allocated(sample.device) / 1024**2


def token_ratio(actions: torch.Tensor, n_max: int | None = None) -> float:
    counts = torch.tensor(TOKEN_COUNTS, device=actions.device, dtype=torch.float32)
    n = counts.index_select(0, actions.flatten().long()).sum()
    total = actions.numel() * (n_max if n_max is not None else max(TOKEN_COUNTS))
    return float((n / max(total, 1)).item())
