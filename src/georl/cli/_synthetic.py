from __future__ import annotations

from collections.abc import Iterator

import torch

from georl.configs.schemas import ExperimentConfig


def synthetic_batches(cfg: ExperimentConfig) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator()
    g.manual_seed(cfg.train.seed + 7)
    while True:
        image = torch.randn(
            cfg.data.batch_size,
            cfg.model.in_channels,
            cfg.model.image_size,
            cfg.model.image_size,
            generator=g,
        )
        if cfg.model.head == "segmentation":
            target = torch.randint(
                0,
                cfg.model.num_classes,
                (cfg.data.batch_size, cfg.model.image_size, cfg.model.image_size),
                generator=g,
            )
        else:
            target = torch.randint(0, cfg.model.num_classes, (cfg.data.batch_size,), generator=g)
        yield image, target
