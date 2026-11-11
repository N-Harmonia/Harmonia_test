"""GeoRL training loop; Sec. 3.9, Algorithm 1."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from georl.configs.schemas import ExperimentConfig
from georl.losses.consistency import consistency_loss
from georl.losses.ppo import PpoOutput, total_objective
from georl.metrics.efficiency import token_ratio as token_ratio_fn
from georl.rl.actions import TOKEN_COUNTS
from georl.rl.gae import compute_gae
from georl.rl.rewards import RewardWeights, composite_reward, cost_reward, task_reward
from georl.training.amp import AmpContext
from georl.training.checkpoint import CheckpointPayload, save_checkpoint_atomic
from georl.training.optim import build_optimizer
from georl.training.scheduler import build_scheduler
from georl.utils.logging import get_logger
from georl.utils.seed import capture_rng_state, set_seed

TaskLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
HeadApplyFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(slots=True, kw_only=True)
class TrainStepResult:
    step: int
    epoch: int
    loss_total: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float
    avg_reward: float
    token_ratio: float
    lr: float


@dataclass(slots=True)
class _Trajectory:
    images: torch.Tensor
    targets: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    masks_prev: torch.Tensor
    t_over_T: torch.Tensor


class GeoRLTrainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        state_encoder: nn.Module,
        policy: nn.Module,
        value: nn.Module,
        tokenizer: nn.Module,
        backbone: nn.Module,
        head_apply: HeadApplyFn,
        task_loss_fn: TaskLossFn,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.state_encoder = state_encoder.to(device)
        self.policy = policy.to(device)
        self.value = value.to(device)
        self.tokenizer = tokenizer.to(device)
        self.backbone = backbone.to(device)
        self.head_apply = head_apply
        self.task_loss_fn = task_loss_fn
        trainable: list[torch.nn.Parameter] = []
        for m in (self.state_encoder, self.policy, self.value, self.tokenizer):
            trainable += list(m.parameters())
        self.optimizer = build_optimizer(trainable, cfg.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.scheduler)
        self.amp = AmpContext(enabled=cfg.train.amp)
        self.weights = RewardWeights(
            alpha=cfg.reward.alpha,
            beta=cfg.reward.beta,
            lam=cfg.reward.lam,
            n_max=cfg.reward.n_max,
        )
        self._step = 0
        self._epoch = 0
        self._logger = get_logger("georl.train")
        set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    @property
    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        for m in (self.state_encoder, self.policy, self.value, self.tokenizer):
            yield from m.parameters()

    @torch.no_grad()
    def _task_loss_with_actions(self, image: torch.Tensor, actions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tokens, mask = self.tokenizer(image, actions)
        encode = getattr(self.backbone, "encode_tokens", self.backbone)
        encoded = encode(tokens, mask)
        pred = self.head_apply(encoded, mask)
        return self.task_loss_fn(pred, target)

    def _policy_value_forward(
        self,
        image: torch.Tensor,
        mask_prev: torch.Tensor,
        t_over_T: torch.Tensor,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor, torch.Tensor]:
        state = self.state_encoder(image, mask_prev, t_over_T)
        logits = self.policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        value_seq = self.value(state)
        value = value_seq.mean(dim=-1) if value_seq.ndim > 1 else value_seq
        return dist, value, state

    @torch.no_grad()
    def _rollout(self, image: torch.Tensor, target: torch.Tensor, horizon: int) -> _Trajectory:
        b = image.shape[0]
        side_sq = self.cfg.model.num_regions
        side = int(side_sq**0.5)
        mask_prev = torch.zeros((b, side_sq), device=self.device)
        prev_actions: torch.Tensor | None = None
        prev_task_loss = self._task_loss_with_actions(
            image, torch.zeros((b, side_sq), dtype=torch.long, device=self.device), target
        ).expand(b)
        traj_actions: list[torch.Tensor] = []
        traj_logp: list[torch.Tensor] = []
        traj_vals: list[torch.Tensor] = []
        traj_r: list[torch.Tensor] = []
        traj_d: list[torch.Tensor] = []
        traj_masks: list[torch.Tensor] = []
        traj_tT: list[torch.Tensor] = []
        for t in range(horizon):
            t_over_T = torch.full((b,), (t + 1) / horizon, device=self.device)
            dist, value, state = self._policy_value_forward(image, mask_prev, t_over_T)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).mean(dim=-1)
            cur_task_loss = self._task_loss_with_actions(image, actions, target).expand(b)
            r_task = task_reward(prev_task_loss, cur_task_loss)
            r_cost = cost_reward(actions, self.cfg.reward.n_max).mean(dim=-1)
            r_cons = torch.zeros(b, device=self.device)
            if prev_actions is not None:
                for bi in range(b):
                    r_cons[bi] = -consistency_loss(
                        actions[bi], prev_actions[bi], state[bi].detach(), side
                    ).detach()
            reward = composite_reward(r_task, r_cost, r_cons, self.weights)
            done = torch.zeros(b, device=self.device) if t < horizon - 1 else torch.ones(b, device=self.device)
            traj_actions.append(actions)
            traj_logp.append(log_probs)
            traj_vals.append(value)
            traj_r.append(reward)
            traj_d.append(done)
            traj_masks.append(mask_prev.clone())
            traj_tT.append(t_over_T)
            mask_prev = torch.maximum(mask_prev, (actions != 3).float())
            prev_task_loss = cur_task_loss
            prev_actions = actions
        return _Trajectory(
            images=image,
            targets=target,
            actions=torch.stack(traj_actions),
            log_probs=torch.stack(traj_logp),
            values=torch.stack(traj_vals),
            rewards=torch.stack(traj_r),
            dones=torch.stack(traj_d),
            masks_prev=torch.stack(traj_masks),
            t_over_T=torch.stack(traj_tT),
        )

    def _ppo_update(self, traj: _Trajectory) -> PpoOutput:
        adv, ret = compute_gae(
            traj.rewards,
            traj.values,
            traj.dones,
            gamma=self.cfg.reward.gamma,
            lam=self.cfg.reward.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        horizon = traj.actions.shape[0]
        with self.amp.autocast():
            new_log_probs_list: list[torch.Tensor] = []
            new_values_list: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            for t in range(horizon):
                dist, value, _ = self._policy_value_forward(traj.images, traj.masks_prev[t], traj.t_over_T[t])
                new_log_probs_list.append(dist.log_prob(traj.actions[t]).mean(dim=-1))
                new_values_list.append(value)
                entropies.append(dist.entropy().mean(dim=-1))
            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list)
            entropy = torch.stack(entropies)
            out = total_objective(
                new_log_probs=new_log_probs,
                old_log_probs=traj.log_probs,
                advantages=adv,
                new_values=new_values,
                old_values=traj.values,
                returns=ret,
                entropy=entropy,
                clip_eps=self.cfg.reward.clip_eps,
                c_value=self.cfg.reward.value_loss_coef,
                c_entropy=self.cfg.reward.entropy_coef,
            )
        return out

    def step(self, image: torch.Tensor, target: torch.Tensor, horizon: int = 4) -> TrainStepResult:
        image = image.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        traj = self._rollout(image, target, horizon=horizon)
        out = self._ppo_update(traj)
        self.optimizer.zero_grad(set_to_none=True)
        self.amp.scale(out.total).backward()
        self.amp.unscale(self.optimizer)
        torch.nn.utils.clip_grad_norm_(list(self.trainable_parameters), self.cfg.optim.grad_clip)
        self.amp.step(self.optimizer)
        self.scheduler.step()
        self._step += 1
        tr = token_ratio_fn(traj.actions[-1], n_max=max(TOKEN_COUNTS))
        return TrainStepResult(
            step=self._step,
            epoch=self._epoch,
            loss_total=float(out.total.item()),
            policy_loss=float(out.policy_loss.item()),
            value_loss=float(out.value_loss.item()),
            entropy=float(out.entropy.item()),
            approx_kl=float(out.approx_kl.item()),
            clip_frac=float(out.clip_frac.item()),
            avg_reward=float(traj.rewards.mean().item()),
            token_ratio=tr,
            lr=self.scheduler.get_last_lr()[0],
        )

    def fit(
        self,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        max_steps: int,
        horizon: int = 4,
    ) -> list[TrainStepResult]:
        results: list[TrainStepResult] = []
        for image, target in batches:
            r = self.step(image, target, horizon=horizon)
            results.append(r)
            if r.step % self.cfg.train.log_interval == 0:
                self._logger.info(
                    "step=%d loss=%.4f policy=%.4f value=%.4f entropy=%.4f kl=%.4f tr=%.3f lr=%.2e",
                    r.step,
                    r.loss_total,
                    r.policy_loss,
                    r.value_loss,
                    r.entropy,
                    r.approx_kl,
                    r.token_ratio,
                    r.lr,
                )
            if r.step >= max_steps:
                break
        return results

    def save(self, path: Path) -> Path:
        payload = CheckpointPayload(
            step=self._step,
            epoch=self._epoch,
            seed=self.cfg.train.seed,
            model={
                "state_encoder": self.state_encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "tokenizer": self.tokenizer.state_dict(),
            },
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.amp.scaler.state_dict() if self.amp.scaler is not None else None,
            rng=capture_rng_state(),
        )
        return save_checkpoint_atomic(payload, path)
