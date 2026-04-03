"""Paper-style multi-critic PPO for reactive soccer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from mjlab.tasks.soccer_reactive.models.symmetry import SoccerSymmetry
from mjlab.tasks.soccer_reactive.rl.reactive_rollout_storage import (
  ReactiveRolloutStorage,
)


@dataclass
class _LossScales:
  surrogate: float = 0.0
  goal_value: float = 0.0
  aux_value: float = 0.0
  reconstruction: float = 0.0
  symmetry: float = 0.0
  entropy: float = 0.0
  kl: float = 0.0


class ReactiveSoccerPPO:
  """Paper-aligned PPO implementation with dual critics and decoder loss."""

  ACTION_CLIP = 1.0
  ADVANTAGE_CLIP = 10.0
  VALUE_LOSS_BETA = 10.0

  def __init__(
    self,
    device: str = "cpu",
    actor_critic: nn.Module | None = None,
    action_dim: int = 23,
    learning_rate: float = 1.0e-3,
    action_std: float = 0.35,
    gamma: float = 0.995,
    lam: float = 0.95,
    clip_param: float = 0.2,
    value_loss_coef: float = 1.0,
    use_clipped_value_loss: bool = True,
    reconstruction_coef: float = 1.0,
    symmetry_coef: float = 10.0,
    entropy_coef: float = 0.01,
    desired_kl: float = 0.01,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    schedule: str = "adaptive",
    max_grad_norm: float = 1.0,
    w_goal: float = 2.0,
    w_aux: float = 1.0,
    symmetry_helper: SoccerSymmetry | None = None,
  ) -> None:
    self.device = device
    self.actor_critic = actor_critic
    self.action_dim = action_dim
    self.action_std = action_std
    self.gamma = gamma
    self.lam = lam
    self.clip_param = clip_param
    self.value_loss_coef = value_loss_coef
    self.use_clipped_value_loss = use_clipped_value_loss
    self.reconstruction_coef = reconstruction_coef
    self.symmetry_coef = symmetry_coef
    self.entropy_coef = entropy_coef
    self.desired_kl = desired_kl
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.schedule = schedule
    self.max_grad_norm = max_grad_norm
    self.w_goal = w_goal
    self.w_aux = w_aux
    self.symmetry_helper = symmetry_helper
    self.learning_rate = learning_rate
    self.storage: ReactiveRolloutStorage | None = None
    self.loss_scales = _LossScales()
    self._last_obs: dict[str, torch.Tensor] | None = None
    self._last_action_info: dict[str, torch.Tensor] | None = None
    self.optimizer = (
      torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
      if self.actor_critic is not None
      else None
    )

  @classmethod
  def construct_algorithm(
    cls,
    device: str = "cpu",
    **kwargs: Any,
  ) -> "ReactiveSoccerPPO":
    return cls(device=device, **kwargs)

  @staticmethod
  def combine_advantages(
    goal_adv: torch.Tensor,
    aux_adv: torch.Tensor,
    w_goal: float,
    w_aux: float,
  ) -> torch.Tensor:
    return w_goal * goal_adv + w_aux * aux_adv

  def initialize_storage(
    self,
    num_envs: int,
    num_steps: int,
    actor_obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    critic_obs_shape: tuple[int, ...] | None = None,
    reconstruction_shape: tuple[int, ...] = (14,),
    actor_history_shape: tuple[int, ...] | None = None,
    privileged_shape: tuple[int, ...] | None = None,
  ) -> ReactiveRolloutStorage:
    self.storage = ReactiveRolloutStorage(
      num_envs=num_envs,
      num_steps=num_steps,
      actor_obs_shape=actor_obs_shape,
      critic_obs_shape=critic_obs_shape,
      action_shape=action_shape,
      reconstruction_shape=reconstruction_shape,
      actor_history_shape=actor_history_shape,
      privileged_shape=privileged_shape,
      device=self.device,
    )
    return self.storage

  def _dist(self, action_mean: torch.Tensor) -> Normal:
    std = torch.full_like(action_mean, max(float(self.action_std), 1.0e-3))
    return Normal(action_mean, std)

  @staticmethod
  def _unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)

  @staticmethod
  def _sanitize_tensor(
    tensor: torch.Tensor,
    *,
    clip: float | None = None,
  ) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
      tensor = torch.clamp(tensor, min=-clip, max=clip)
    return tensor

  def _distributed_any(self, predicate: bool) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
      return bool(predicate)
    flag = torch.tensor(
      1 if predicate else 0,
      device=self.device,
      dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())

  def _distributed_skip_increment(self, triggered: bool) -> float:
    if not triggered:
      return 0.0
    if not (dist.is_available() and dist.is_initialized()):
      return 1.0
    return 1.0 / float(max(dist.get_world_size(), 1))

  def act(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    self._last_obs = obs
    if self.actor_critic is None:
      batch = obs["actor_current"].shape[0]
      return {"actions": torch.zeros((batch, self.action_dim), device=self.device)}
    with torch.no_grad():
      outputs = self.actor_critic(obs)
      dist = self._dist(outputs["actions_mean"])
      actions = self._sanitize_tensor(dist.sample(), clip=self.ACTION_CLIP)
      log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
    self._last_action_info = {
      "actions": self._sanitize_tensor(actions, clip=self.ACTION_CLIP),
      "action_mean": self._sanitize_tensor(
        outputs["actions_mean"],
        clip=self.ACTION_CLIP,
      ),
      "old_log_probs": self._sanitize_tensor(log_probs),
      "goal_value": self._sanitize_tensor(outputs["goal_value"]),
      "aux_value": self._sanitize_tensor(outputs["aux_value"]),
      "reconstruction": self._sanitize_tensor(outputs["reconstruction"]),
    }
    return {"actions": actions}

  def process_env_step(
    self,
    obs: dict[str, torch.Tensor],
    rewards: torch.Tensor | dict[str, torch.Tensor] | None,
    dones: torch.Tensor | None,
    extras: dict[str, Any] | None,
  ) -> None:
    del extras
    self._last_obs = obs
    if self.storage is None or self._last_action_info is None or rewards is None:
      return

    if isinstance(rewards, dict):
      goal_rewards = rewards.get("goal", rewards.get("total"))
      aux_rewards = rewards.get("aux", rewards.get("total"))
    else:
      goal_rewards = rewards
      aux_rewards = rewards

    assert goal_rewards is not None
    assert aux_rewards is not None
    goal_rewards = self._sanitize_tensor(goal_rewards)
    aux_rewards = self._sanitize_tensor(aux_rewards)
    if dones is None:
      dones = torch.zeros(
        goal_rewards.shape[0], device=goal_rewards.device, dtype=torch.long
      )
    dones = self._sanitize_tensor(dones.float()).to(dtype=torch.float32)

    self.storage.add(
      actor_current=self._sanitize_tensor(obs["actor_current"]),
      critic_current=(
        None
        if obs.get("critic_current") is None
        else self._sanitize_tensor(obs["critic_current"])
      ),
      actor_history=(
        None
        if obs.get("actor_history") is None
        else self._sanitize_tensor(obs["actor_history"])
      ),
      critic_privileged=(
        None
        if obs.get("critic_privileged") is None
        else self._sanitize_tensor(obs["critic_privileged"])
      ),
      actions=self._last_action_info["actions"],
      old_action_mean=self._last_action_info["action_mean"],
      old_log_probs=self._last_action_info["old_log_probs"],
      goal_values=self._last_action_info["goal_value"],
      aux_values=self._last_action_info["aux_value"],
      goal_rewards=goal_rewards.unsqueeze(-1)
      if goal_rewards.ndim == 1
      else goal_rewards,
      aux_rewards=aux_rewards.unsqueeze(-1) if aux_rewards.ndim == 1 else aux_rewards,
      dones=dones.unsqueeze(-1).float() if dones.ndim == 1 else dones.float(),
      reconstruction_targets=self._sanitize_tensor(obs["critic_privileged"]),
    )

  def compute_returns(self, obs: dict[str, torch.Tensor] | None = None) -> None:
    if obs is not None:
      self._last_obs = obs
    if self.storage is None or self.actor_critic is None or self._last_obs is None:
      return
    with torch.no_grad():
      next_outputs = self.actor_critic(self._last_obs)
    self.storage.compute_returns(
      next_goal_value=self._sanitize_tensor(next_outputs["goal_value"]),
      next_aux_value=self._sanitize_tensor(next_outputs["aux_value"]),
      gamma=self.gamma,
      lam=self.lam,
    )

  def _iter_minibatches(self, batch: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    batch_size = batch["actor_current"].shape[0]
    if batch_size == 0:
      return []
    permutation = torch.randperm(batch_size, device=self.device)
    chunks = torch.chunk(permutation, self.num_mini_batches)
    mini_batches: list[dict[str, torch.Tensor]] = []
    for ids in chunks:
      if ids.numel() == 0:
        continue
      mini_batches.append({key: value[ids] for key, value in batch.items()})
    return mini_batches

  def _compute_kl(
    self,
    current_mean: torch.Tensor,
    old_mean: torch.Tensor,
  ) -> torch.Tensor:
    variance = self.action_std**2
    return ((current_mean - old_mean).pow(2) / (2.0 * variance)).mean()

  def _value_loss(
    self,
    predicted_value: torch.Tensor,
    old_value: torch.Tensor,
    target_value: torch.Tensor,
  ) -> torch.Tensor:
    predicted_value = self._sanitize_tensor(predicted_value)
    old_value = self._sanitize_tensor(old_value)
    target_value = self._sanitize_tensor(target_value)
    value_loss = F.smooth_l1_loss(
      predicted_value,
      target_value,
      reduction="none",
      beta=self.VALUE_LOSS_BETA,
    )
    if not self.use_clipped_value_loss:
      return value_loss.mean()
    clipped_value = old_value + (predicted_value - old_value).clamp(
      min=-self.clip_param,
      max=self.clip_param,
    )
    clipped_value_loss = F.smooth_l1_loss(
      clipped_value,
      target_value,
      reduction="none",
      beta=self.VALUE_LOSS_BETA,
    )
    return torch.max(value_loss, clipped_value_loss).mean()

  def _maybe_adapt_learning_rate(self, mean_kl: float) -> None:
    if self.optimizer is None or self.schedule != "adaptive" or self.desired_kl <= 0.0:
      return
    if mean_kl > 2.0 * self.desired_kl:
      self.learning_rate = max(self.learning_rate / 1.5, 1.0e-5)
    elif 0.0 < mean_kl < 0.5 * self.desired_kl:
      self.learning_rate = min(self.learning_rate * 1.5, 1.0e-2)
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = self.learning_rate

  def update(self) -> dict[str, float]:
    if (
      self.actor_critic is None
      or self.optimizer is None
      or self.storage is None
      or self.storage.step == 0
    ):
      return {
        "surrogate": self.loss_scales.surrogate,
        "goal_value": self.loss_scales.goal_value,
        "aux_value": self.loss_scales.aux_value,
        "reconstruction": self.loss_scales.reconstruction,
        "symmetry": self.loss_scales.symmetry,
        "entropy": self.loss_scales.entropy,
        "kl": self.loss_scales.kl,
        "goal_value_abs_mean": 0.0,
        "goal_value_abs_max": 0.0,
        "aux_value_abs_mean": 0.0,
        "aux_value_abs_max": 0.0,
        "goal_return_abs_mean": 0.0,
        "goal_return_abs_max": 0.0,
        "aux_return_abs_mean": 0.0,
        "aux_return_abs_max": 0.0,
        "skipped_nonfinite_loss_updates": 0.0,
        "skipped_nonfinite_grad_updates": 0.0,
      }

    batch = self.storage.flattened_batch()
    aggregate = {
      "surrogate": 0.0,
      "goal_value": 0.0,
      "aux_value": 0.0,
      "reconstruction": 0.0,
      "symmetry": 0.0,
      "entropy": 0.0,
      "kl": 0.0,
      "goal_value_abs_mean": 0.0,
      "aux_value_abs_mean": 0.0,
      "goal_return_abs_mean": 0.0,
      "aux_return_abs_mean": 0.0,
      "skipped_nonfinite_loss_updates": 0.0,
      "skipped_nonfinite_grad_updates": 0.0,
    }
    aggregate_max = {
      "goal_value_abs_max": 0.0,
      "aux_value_abs_max": 0.0,
      "goal_return_abs_max": 0.0,
      "aux_return_abs_max": 0.0,
    }
    num_updates = 0

    for _epoch in range(self.num_learning_epochs):
      for mini_batch in self._iter_minibatches(batch):
        outputs = self.actor_critic(
          {
            "actor_current": mini_batch["actor_current"],
            "critic_current": mini_batch["critic_current"],
            "actor_history": mini_batch["actor_history"],
            "critic_privileged": mini_batch["critic_privileged"],
          }
        )
        dist = self._dist(outputs["actions_mean"])
        log_probs = self._sanitize_tensor(
          dist.log_prob(mini_batch["actions"]).sum(dim=-1, keepdim=True)
        )
        goal_value_abs = outputs["goal_value"].detach().abs()
        aux_value_abs = outputs["aux_value"].detach().abs()
        goal_return_abs = mini_batch["goal_returns"].detach().abs()
        aux_return_abs = mini_batch["aux_returns"].detach().abs()
        aggregate["goal_value_abs_mean"] += float(goal_value_abs.mean().item())
        aggregate["aux_value_abs_mean"] += float(aux_value_abs.mean().item())
        aggregate["goal_return_abs_mean"] += float(goal_return_abs.mean().item())
        aggregate["aux_return_abs_mean"] += float(aux_return_abs.mean().item())
        aggregate_max["goal_value_abs_max"] = max(
          aggregate_max["goal_value_abs_max"],
          float(goal_value_abs.max().item()),
        )
        aggregate_max["aux_value_abs_max"] = max(
          aggregate_max["aux_value_abs_max"],
          float(aux_value_abs.max().item()),
        )
        aggregate_max["goal_return_abs_max"] = max(
          aggregate_max["goal_return_abs_max"],
          float(goal_return_abs.max().item()),
        )
        aggregate_max["aux_return_abs_max"] = max(
          aggregate_max["aux_return_abs_max"],
          float(aux_return_abs.max().item()),
        )

        total_adv = self.combine_advantages(
          mini_batch["goal_advantages"],
          mini_batch["aux_advantages"],
          w_goal=self.w_goal,
          w_aux=self.w_aux,
        )
        total_adv = self._sanitize_tensor(total_adv)
        total_adv = (total_adv - total_adv.mean()) / total_adv.std(
          unbiased=False
        ).clamp(min=1.0e-6)
        total_adv = torch.clamp(
          total_adv,
          min=-self.ADVANTAGE_CLIP,
          max=self.ADVANTAGE_CLIP,
        )
        ratio = torch.exp(
          torch.clamp(
            log_probs - mini_batch["old_log_probs"],
            min=-20.0,
            max=20.0,
          )
        )
        surrogate_1 = ratio * total_adv
        surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * total_adv
        surrogate_loss = -torch.min(surrogate_1, surrogate_2).mean()

        goal_value_loss = self._value_loss(
          outputs["goal_value"],
          mini_batch["goal_values"],
          mini_batch["goal_returns"],
        )
        aux_value_loss = self._value_loss(
          outputs["aux_value"],
          mini_batch["aux_values"],
          mini_batch["aux_returns"],
        )
        reconstruction_loss = F.mse_loss(
          outputs["reconstruction"], mini_batch["reconstruction_targets"]
        )
        symmetry_loss = torch.zeros((), device=self.device)
        if self.symmetry_helper is not None:
          mirrored_outputs = self.actor_critic(
            {
              "actor_current": self.symmetry_helper.mirror_actor_obs(
                mini_batch["actor_current"]
              ),
              "critic_current": self.symmetry_helper.mirror_actor_obs(
                mini_batch["critic_current"]
              ),
              "actor_history": self.symmetry_helper.mirror_actor_history(
                mini_batch["actor_history"]
              ),
              "critic_privileged": mini_batch["critic_privileged"],
            }
          )
          symmetry_loss = self.symmetry_helper.symmetry_loss(
            outputs["actions_mean"], mirrored_outputs["actions_mean"]
          )
        entropy_loss = -dist.entropy().sum(dim=-1).mean()
        approx_kl = self._compute_kl(
          outputs["actions_mean"], mini_batch["old_action_mean"]
        )
        approx_kl = self._sanitize_tensor(approx_kl)

        loss = (
          surrogate_loss
          + self.value_loss_coef * (goal_value_loss + aux_value_loss)
          + self.reconstruction_coef * reconstruction_loss
          + self.symmetry_coef * symmetry_loss
          + self.entropy_coef * entropy_loss
        )
        loss = self._sanitize_tensor(loss)
        skip_loss_update = self._distributed_any(
          not bool(torch.isfinite(loss).item())
        )
        if skip_loss_update:
          aggregate["skipped_nonfinite_loss_updates"] += self._distributed_skip_increment(
            True
          )
          self.optimizer.zero_grad(set_to_none=True)
          continue
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        has_nonfinite_grad = False
        for param in self.actor_critic.parameters():
          if param.grad is None:
            continue
          if not torch.all(torch.isfinite(param.grad)):
            has_nonfinite_grad = True
            break
        skip_grad_update = self._distributed_any(has_nonfinite_grad)
        if skip_grad_update:
          aggregate["skipped_nonfinite_grad_updates"] += self._distributed_skip_increment(
            True
          )
          self.optimizer.zero_grad(set_to_none=True)
          continue
        torch.nn.utils.clip_grad_norm_(
          self.actor_critic.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()

        aggregate["surrogate"] += float(surrogate_loss.detach().item())
        aggregate["goal_value"] += float(goal_value_loss.detach().item())
        aggregate["aux_value"] += float(aux_value_loss.detach().item())
        aggregate["reconstruction"] += float(reconstruction_loss.detach().item())
        aggregate["symmetry"] += float(symmetry_loss.detach().item())
        aggregate["entropy"] += float((-entropy_loss).detach().item())
        aggregate["kl"] += float(approx_kl.detach().item())
        num_updates += 1

    self.storage.reset()
    mean_keys = (
      "surrogate",
      "goal_value",
      "aux_value",
      "reconstruction",
      "symmetry",
      "entropy",
      "kl",
      "goal_value_abs_mean",
      "aux_value_abs_mean",
      "goal_return_abs_mean",
      "aux_return_abs_mean",
    )
    denom = max(num_updates, 1)
    averaged = {
      key: (aggregate[key] / denom if key in mean_keys else aggregate[key])
      for key in aggregate
    }
    averaged.update(aggregate_max)
    self._maybe_adapt_learning_rate(averaged["kl"])
    self.loss_scales = _LossScales(
      surrogate=averaged["surrogate"],
      goal_value=averaged["goal_value"],
      aux_value=averaged["aux_value"],
      reconstruction=averaged["reconstruction"],
      symmetry=averaged["symmetry"],
      entropy=averaged["entropy"],
      kl=averaged["kl"],
    )
    averaged["learning_rate"] = self.learning_rate
    return averaged

  def save(self) -> dict[str, Any]:
    state: dict[str, Any] = {"device": self.device, "learning_rate": self.learning_rate}
    if self.actor_critic is not None:
      state["actor_critic"] = self._unwrap_module(self.actor_critic).state_dict()
    if self.optimizer is not None:
      state["optimizer"] = self.optimizer.state_dict()
    return state

  def load(self, state: dict[str, Any]) -> None:
    self.device = state.get("device", self.device)
    self.learning_rate = state.get("learning_rate", self.learning_rate)
    if self.actor_critic is not None and "actor_critic" in state:
      self._unwrap_module(self.actor_critic).load_state_dict(state["actor_critic"])
    if self.optimizer is not None and "optimizer" in state:
      self.optimizer.load_state_dict(state["optimizer"])

  def get_inference_policy(
    self,
  ) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    def _policy(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
      if self.actor_critic is None:
        return self.act(obs)
      with torch.no_grad():
        outputs = self.actor_critic(obs)
      return {"actions": outputs["actions_mean"]}

    return _policy
