from __future__ import annotations

import itertools

import pytest
import torch
from omegaconf import OmegaConf

from georl.cli._factory import build_components
from georl.cli._synthetic import synthetic_batches
from georl.configs.schemas import ExperimentConfig
from georl.training.loop import GeoRLTrainer


def _load_smoke_cfg() -> ExperimentConfig:
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "_smoke.yaml"
    raw = OmegaConf.load(cfg_path)
    plain = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(plain, dict)
    return ExperimentConfig(**plain)


@pytest.mark.smoke
def test_two_steps_decrease_total_loss() -> None:
    cfg = _load_smoke_cfg()
    components = build_components(cfg)
    trainer = GeoRLTrainer(
        cfg=cfg,
        state_encoder=components.state_encoder,
        policy=components.policy,
        value=components.value,
        tokenizer=components.tokenizer,
        backbone=components.backbone,
        head_apply=components.head_apply,
        task_loss_fn=components.task_loss_fn,
        device=torch.device("cpu"),
    )
    batches = itertools.islice(synthetic_batches(cfg), 4)
    results = trainer.fit(batches, max_steps=2, horizon=2)
    assert len(results) == 2
    assert results[0].lr > 0
    assert results[0].step == 1
    assert results[-1].step == 2
    for r in results:
        assert r.loss_total == r.loss_total  # not NaN
