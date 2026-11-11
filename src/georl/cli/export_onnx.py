from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from georl.cli._factory import build_components
from georl.configs.schemas import ExperimentConfig, register_configs
from georl.utils.logging import configure_logging, get_logger

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")

register_configs()


class _PolicyOnly(torch.nn.Module):
    def __init__(self, state_encoder: torch.nn.Module, policy: torch.nn.Module) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.policy = policy

    def forward(self, image: torch.Tensor, mask_prev: torch.Tensor, t_over_T: torch.Tensor) -> torch.Tensor:
        state = self.state_encoder(image, mask_prev, t_over_T)
        return self.policy(state).argmax(dim=-1)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="experiment/main_dota")
def main(raw_cfg: DictConfig) -> None:
    configure_logging("INFO")
    logger = get_logger("georl.cli.export")
    cfg_obj = OmegaConf.to_object(raw_cfg)
    cfg: ExperimentConfig = cfg_obj if isinstance(cfg_obj, ExperimentConfig) else ExperimentConfig(**OmegaConf.to_container(raw_cfg, resolve=True))  # type: ignore[arg-type]
    comp = build_components(cfg)
    wrapper = _PolicyOnly(comp.state_encoder, comp.policy).eval()
    out_dir = Path("outputs") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dummy_image = torch.randn(1, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size)
    dummy_mask = torch.zeros(1, cfg.model.num_regions)
    dummy_t = torch.tensor([1.0])
    onnx_path = out_dir / "policy.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_mask, dummy_t),
        onnx_path,
        input_names=["image", "mask_prev", "t_over_T"],
        output_names=["actions"],
        dynamic_axes={"image": {0: "batch"}, "mask_prev": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    logger.info("exported ONNX: %s", onnx_path)


if __name__ == "__main__":
    main()
