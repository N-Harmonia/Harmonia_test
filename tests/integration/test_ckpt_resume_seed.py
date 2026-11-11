from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from georl.training.checkpoint import CheckpointPayload, load_checkpoint, save_checkpoint_atomic
from georl.utils.seed import capture_rng_state, set_seed


@pytest.mark.integration
def test_atomic_save_and_round_trip(tmp_path: Path) -> None:
    set_seed(11)
    model = nn.Linear(4, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optim.zero_grad()
    (model(torch.zeros(1, 4)) ** 2).sum().backward()
    optim.step()
    payload = CheckpointPayload(
        step=10,
        epoch=1,
        seed=11,
        model={"linear": model.state_dict()},
        optimizer=optim.state_dict(),
        rng=capture_rng_state(),
    )
    dest = tmp_path / "ckpt.pt"
    save_checkpoint_atomic(payload, dest)
    loaded = load_checkpoint(dest)
    assert loaded.step == 10
    assert loaded.seed == 11
    assert loaded.rng is not None
    new_model = nn.Linear(4, 2)
    new_model.load_state_dict(loaded.model["linear"])
    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.allclose(p1, p2)


@pytest.mark.integration
def test_no_partial_file_left_on_disk(tmp_path: Path) -> None:
    set_seed(3)
    payload = CheckpointPayload(
        step=1,
        epoch=0,
        seed=3,
        model={"x": {"w": torch.zeros(2)}},
        optimizer={},
    )
    save_checkpoint_atomic(payload, tmp_path / "a.pt")
    stragglers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert stragglers == []
