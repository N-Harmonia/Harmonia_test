from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 0.5


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 500
    total_steps: int = 100000
    min_lr_factor: float = 0.01


@dataclass
class RewardConfig:
    alpha: float = 1.0
    beta: float = 0.3
    lam: float = 0.1
    n_max: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    minibatch_size: int = 256


@dataclass
class ModelConfig:
    state_encoder_dim: int = 256
    transformer_blocks: int = 4
    transformer_heads: int = 4
    transformer_ffn: int = 1024
    transformer_dropout: float = 0.1
    base_patch: int = 16
    num_regions: int = 1024
    image_size: int = 512
    vfm: str = "vit_l14"
    num_classes: int = 15
    head: str = "detection"
    in_channels: int = 3


@dataclass
class DataConfig:
    name: str = "dota"
    root: str = "data/dota"
    split: str = "train"
    num_workers: int = 4
    batch_size: int = 8
    image_size: int = 512
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 50
    iterations_per_epoch: int = 1000
    seed: int = 42
    amp: bool = True
    grad_accum: int = 1
    log_interval: int = 50
    eval_interval: int = 1000
    ckpt_dir: str = "ckpts"
    ckpt_keep_last: int = 3
    ema_decay: float = 0.0
    deterministic: bool = False


_SECTION_TYPES: dict[str, type] = {}


@dataclass
class ExperimentConfig:
    name: str = "main_dota"
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if isinstance(self.optim, dict):
            self.optim = OptimConfig(**self.optim)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)
        if isinstance(self.reward, dict):
            self.reward = RewardConfig(**self.reward)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            extra = self.data.get("extra", {}) if isinstance(self.data, dict) else {}
            data = {**self.data, "extra": dict(extra)}
            self.data = DataConfig(**data)
        if isinstance(self.train, dict):
            self.train = TrainConfig(**self.train)
        if isinstance(self.optim.betas, list):
            self.optim.betas = (float(self.optim.betas[0]), float(self.optim.betas[1]))


def register_configs() -> None:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="experiment_schema", node=ExperimentConfig)
    cs.store(group="model", name="defaults", node=ModelConfig)
    cs.store(group="data", name="defaults", node=DataConfig)
    cs.store(group="train", name="defaults", node=TrainConfig)
