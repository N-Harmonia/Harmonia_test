from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from georl.cli._factory import build_components
from georl.configs.schemas import ExperimentConfig, register_configs
from georl.evaluation.inference import run_inference
from georl.training.checkpoint import load_checkpoint
from georl.utils.logging import configure_logging, get_logger

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")

register_configs()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="experiment/main_dota")
def main(raw_cfg: DictConfig) -> None:
    configure_logging("INFO")
    logger = get_logger("georl.cli.infer")
    cfg_obj = OmegaConf.to_object(raw_cfg)
    cfg: ExperimentConfig = cfg_obj if isinstance(cfg_obj, ExperimentConfig) else ExperimentConfig(**OmegaConf.to_container(raw_cfg, resolve=True))  # type: ignore[arg-type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comp = build_components(cfg)
    ckpt_path = Path("outputs") / cfg.name / "last.pt"
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path)
        comp.state_encoder.load_state_dict(ck.model["state_encoder"])
        comp.policy.load_state_dict(ck.model["policy"])
        comp.value.load_state_dict(ck.model["value"])
        comp.tokenizer.load_state_dict(ck.model["tokenizer"])
        logger.info("loaded checkpoint: %s (step=%d)", ckpt_path, ck.step)
    image = torch.randn(1, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size, device=device)
    result = run_inference(
        image,
        state_encoder=comp.state_encoder.to(device),
        policy=comp.policy.to(device),
        tokenizer=comp.tokenizer.to(device),
        backbone=comp.backbone.to(device),
        head_apply=comp.head_apply,
        horizon=1,
        num_regions=cfg.model.num_regions,
    )
    out_dir = Path("outputs") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "infer.json").write_text(
        json.dumps(
            {
                "name": cfg.name,
                "token_count": result.token_count,
                "action_histogram": torch.bincount(result.actions.flatten(), minlength=4).tolist(),
                "prediction_shape": list(result.prediction.shape),
            },
            indent=2,
        )
    )
    logger.info("token_count=%d", result.token_count)


if __name__ == "__main__":
    main()
