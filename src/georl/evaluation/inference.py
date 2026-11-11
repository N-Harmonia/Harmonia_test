"""Reconstructed inference path (Algorithm 2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

HeadApplyFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(slots=True, kw_only=True)
class InferenceResult:
    actions: torch.Tensor
    prediction: torch.Tensor
    token_count: int


@torch.no_grad()
def run_inference(
    image: torch.Tensor,
    *,
    state_encoder: nn.Module,
    policy: nn.Module,
    tokenizer: nn.Module,
    backbone: nn.Module,
    head_apply: HeadApplyFn,
    horizon: int,
    num_regions: int,
) -> InferenceResult:
    state_encoder.eval()
    policy.eval()
    tokenizer.eval()
    backbone.eval()
    b = image.shape[0]
    device = image.device
    mask_prev = torch.zeros((b, num_regions), device=device)
    final_actions = torch.zeros((b, num_regions), dtype=torch.long, device=device)
    for t in range(horizon):
        t_over_T = torch.full((b,), (t + 1) / horizon, device=device)
        state = state_encoder(image, mask_prev, t_over_T)
        logits = policy(state)
        actions = logits.argmax(dim=-1)
        final_actions = actions
        mask_prev = torch.maximum(mask_prev, (actions != 3).float())
    tokens, mask = tokenizer(image, final_actions)
    mask_tensor: torch.Tensor = mask
    encode = getattr(backbone, "encode_tokens", backbone)
    encoded = encode(tokens, mask_tensor)
    prediction = head_apply(encoded, mask_tensor)
    token_count = int(mask_tensor.sum().item())
    return InferenceResult(actions=final_actions, prediction=prediction, token_count=token_count)
