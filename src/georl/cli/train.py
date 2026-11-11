from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from georl.cli._factory import build_components
from georl.cli._synthetic import synthetic_batches
from georl.configs.schemas import ExperimentConfig, register_configs
from georl.training.ddp import setup_ddp, teardown_ddp
from georl.training.loop import GeoRLTrainer
from georl.utils.logging import configure_logging, get_logger

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


register_configs()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="experiment/main_dota")
def main(raw_cfg: DictConfig) -> None:
    configure_logging("INFO")
    logger = get_logger("georl.cli.train")
    cfg = OmegaConf.to_object(raw_cfg)
    if not isinstance(cfg, ExperimentConfig):
        cfg = ExperimentConfig(**OmegaConf.to_container(raw_cfg, resolve=True))  # type: ignore[arg-type]
    device = setup_ddp()
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
        device=device,
    )
    max_steps = cfg.train.iterations_per_epoch * cfg.train.epochs
    results = trainer.fit(synthetic_batches(cfg), max_steps=max_steps)
    out_dir = Path("outputs") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": cfg.name,
        "steps": len(results),
        "last": {
            "loss_total": results[-1].loss_total if results else None,
            "policy_loss": results[-1].policy_loss if results else None,
            "value_loss": results[-1].value_loss if results else None,
            "token_ratio": results[-1].token_ratio if results else None,
            "lr": results[-1].lr if results else None,
        },
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    ckpt = trainer.save(out_dir / "last.pt")
    logger.info("saved checkpoint: %s", ckpt)
    teardown_ddp()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
