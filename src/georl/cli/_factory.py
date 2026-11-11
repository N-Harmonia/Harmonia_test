from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from georl.configs.schemas import ExperimentConfig
from georl.models.adaptive_tokenizer import AdaptiveTokenizer
from georl.models.backbones import VfmAdapter, build_vfm
from georl.models.heads import ClassificationHead, DetectionHead, SegmentationHead
from georl.models.policy_net import PolicyNetwork
from georl.models.state_encoder import StateEncoder
from georl.models.value_net import ValueNetwork

HeadApplyFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TaskLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class GeoRLComponents:
    state_encoder: StateEncoder
    policy: PolicyNetwork
    value: ValueNetwork
    tokenizer: AdaptiveTokenizer
    backbone: VfmAdapter
    head: nn.Module
    head_apply: HeadApplyFn
    task_loss_fn: TaskLossFn


def _classification_factory(num_classes: int, embed_dim: int) -> tuple[nn.Module, HeadApplyFn, TaskLossFn]:
    head = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def apply(encoded: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return head(encoded)

    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(pred, target.long())

    return head, apply, loss


def _segmentation_factory(num_classes: int, embed_dim: int, out_size: int) -> tuple[nn.Module, HeadApplyFn, TaskLossFn]:
    head = SegmentationHead(embed_dim=embed_dim, num_classes=num_classes, out_size=out_size)

    def apply(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return head(encoded, mask)

    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tgt = target.squeeze(1) if target.ndim == 4 else target
        return F.cross_entropy(pred, tgt.long(), ignore_index=255)

    return head, apply, loss


def _detection_factory(num_classes: int, embed_dim: int) -> tuple[nn.Module, HeadApplyFn, TaskLossFn]:
    head = DetectionHead(embed_dim=embed_dim, num_classes=num_classes)

    def apply(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cls_logits, _ = head(encoded, mask)
        return cls_logits

    def loss(pred: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        b, q, c = pred.shape
        flat = pred.reshape(b * q, c)
        target_flat = torch.full((b * q,), c - 1, device=pred.device, dtype=torch.long)
        return F.cross_entropy(flat, target_flat)

    return head, apply, loss


def build_components(cfg: ExperimentConfig) -> GeoRLComponents:
    state_encoder = StateEncoder(
        in_channels=cfg.model.in_channels,
        num_regions=cfg.model.num_regions,
        embed_dim=cfg.model.state_encoder_dim,
    )
    policy = PolicyNetwork(
        state_dim=cfg.model.state_encoder_dim,
        num_actions=4,
        num_blocks=cfg.model.transformer_blocks,
        heads=cfg.model.transformer_heads,
        ffn_dim=cfg.model.transformer_ffn,
        dropout=cfg.model.transformer_dropout,
    )
    value = ValueNetwork(
        state_dim=cfg.model.state_encoder_dim,
        num_blocks=cfg.model.transformer_blocks,
        heads=cfg.model.transformer_heads,
        ffn_dim=cfg.model.transformer_ffn,
        dropout=cfg.model.transformer_dropout,
    )
    backbone = build_vfm(cfg.model.vfm)
    backbone.freeze()
    tokenizer = AdaptiveTokenizer(
        in_channels=cfg.model.in_channels,
        base_patch=cfg.model.base_patch,
        embed_dim=backbone.embed_dim,
    )
    if cfg.model.head == "segmentation":
        head, apply, loss = _segmentation_factory(cfg.model.num_classes, backbone.embed_dim, cfg.model.image_size)
    elif cfg.model.head == "detection":
        head, apply, loss = _detection_factory(cfg.model.num_classes, backbone.embed_dim)
    else:
        head, apply, loss = _classification_factory(cfg.model.num_classes, backbone.embed_dim)
    return GeoRLComponents(
        state_encoder=state_encoder,
        policy=policy,
        value=value,
        tokenizer=tokenizer,
        backbone=backbone,
        head=head,
        head_apply=apply,
        task_loss_fn=loss,
    )
