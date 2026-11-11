from __future__ import annotations

import math

import pytest
import torch

from georl.losses.ppo import ppo_clip_loss, total_objective


@pytest.mark.unit
def test_ratio_equal_one_yields_negative_advantage_mean() -> None:
    n = 8
    new_lp = torch.zeros(n)
    old_lp = torch.zeros(n)
    adv = torch.linspace(-1, 1, n)
    loss, kl, cf = ppo_clip_loss(new_lp, old_lp, adv, clip_eps=0.2)
    assert torch.isclose(loss, -adv.mean())
    assert torch.isclose(kl, torch.tensor(0.0))
    assert cf.item() == 0.0


@pytest.mark.unit
def test_clip_kicks_in_when_ratio_exceeds_bounds() -> None:
    new_lp = torch.tensor([math.log(2.0)])
    old_lp = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss, _, cf = ppo_clip_loss(new_lp, old_lp, adv, clip_eps=0.2)
    assert torch.isclose(loss, -torch.tensor(1.2))
    assert cf.item() == 1.0


@pytest.mark.unit
def test_total_objective_includes_entropy_bonus_with_correct_sign() -> None:
    new_lp = torch.zeros(4)
    old_lp = torch.zeros(4)
    adv = torch.zeros(4)
    new_v = torch.zeros(4)
    old_v = torch.zeros(4)
    ret = torch.zeros(4)
    entropy = torch.tensor([1.0, 1.0, 1.0, 1.0])
    out = total_objective(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        advantages=adv,
        new_values=new_v,
        old_values=old_v,
        returns=ret,
        entropy=entropy,
        clip_eps=0.2,
        c_value=0.5,
        c_entropy=0.01,
    )
    assert torch.isclose(out.total, torch.tensor(-0.01))
