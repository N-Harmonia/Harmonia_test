from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    torch.manual_seed(0)
