"""Paper-style multi-critic PPO for reactive soccer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
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

  def __init__(
    self,
    device: str = "cpu",
    actor_critic: nn.Module | None = None,
    action_dim: int = 23,
    learning_rate: float = 1.0e-3,
    action_std: float = 1.0,
    gamma: float = 0.995,
    lam: float = 0.95,
    clip_param: float = 0.2,
    value_loss_coef: float = 1.0,
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
    std = torch.full_like(action_mean, self.action_std)
    return Normal(action_mean, std)

  def act(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    self._last_obs = obs
    if self.actor_critic is None:
      batch = obs["actor_current"].shape[0]
      return {"actions": torch.zeros((batch, self.action_dim), device=self.device)}
    with torch.no_grad():
      outputs = self.actor_critic.forward_train(obs)
      dist = self._dist(outputs["actions_mean"])
      actions = dist.sample()
      log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
    self._last_action_info = {
      "actions": actions,
      "action_mean": outputs["actions_mean"],
      "old_log_probs": log_probs,
      "goal_value": outputs["goal_value"],
      "aux_value": outputs["aux_value"],
      "reconstruction": outputs["reconstruction"],
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
    if dones is None:
      dones = torch.zeros(
        goal_rewards.shape[0], device=goal_rewards.device, dtype=torch.long
      )

    self.storage.add(
      actor_current=obs["actor_current"],
      critic_current=obs.get("critic_current"),
      actor_history=obs.get("actor_history"),
      critic_privileged=obs.get("critic_privileged"),
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
      reconstruction_targets=obs["critic_privileged"],
    )

  def compute_returns(self, obs: dict[str, torch.Tensor] | None = None) -> None:
    if obs is not None:
      self._last_obs = obs
    if self.storage is None or self.actor_critic is None or self._last_obs is None:
      return
    with torch.no_grad():
      next_outputs = self.actor_critic.forward_train(self._last_obs)
    self.storage.compute_returns(
      next_goal_value=next_outputs["goal_value"],
      next_aux_value=next_outputs["aux_value"],
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
    }
    num_updates = 0

    for _epoch in range(self.num_learning_epochs):
      for mini_batch in self._iter_minibatches(batch):
        outputs = self.actor_critic.forward_train(
          {
            "actor_current": mini_batch["actor_current"],
            "critic_current": mini_batch["critic_current"],
            "actor_history": mini_batch["actor_history"],
            "critic_privileged": mini_batch["critic_privileged"],
          }
        )
        dist = self._dist(outputs["actions_mean"])
        log_probs = dist.log_prob(mini_batch["actions"]).sum(dim=-1, keepdim=True)

        total_adv = self.combine_advantages(
          mini_batch["goal_advantages"],
          mini_batch["aux_advantages"],
          w_goal=self.w_goal,
          w_aux=self.w_aux,
        )
        total_adv = (total_adv - total_adv.mean()) / total_adv.std(
          unbiased=False
        ).clamp(min=1.0e-6)
        ratio = torch.exp(log_probs - mini_batch["old_log_probs"])
        surrogate_1 = ratio * total_adv
        surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * total_adv
        surrogate_loss = -torch.min(surrogate_1, surrogate_2).mean()

        goal_value_loss = F.mse_loss(outputs["goal_value"], mini_batch["goal_returns"])
        aux_value_loss = F.mse_loss(outputs["aux_value"], mini_batch["aux_returns"])
        reconstruction_loss = F.mse_loss(
          outputs["reconstruction"], mini_batch["reconstruction_targets"]
        )
        symmetry_loss = torch.zeros((), device=self.device)
        if self.symmetry_helper is not None:
          mirrored_outputs = self.actor_critic.forward_train(
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

        loss = (
          surrogate_loss
          + self.value_loss_coef * (goal_value_loss + aux_value_loss)
          + self.reconstruction_coef * reconstruction_loss
          + self.symmetry_coef * symmetry_loss
          + self.entropy_coef * entropy_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
    if num_updates == 0:
      num_updates = 1
    averaged = {key: value / num_updates for key, value in aggregate.items()}
    self._maybe_adapt_learning_rate(averaged["kl"])
    self.loss_scales = _LossScales(**averaged)
    return averaged

  def save(self) -> dict[str, Any]:
    state: dict[str, Any] = {"device": self.device, "learning_rate": self.learning_rate}
    if self.actor_critic is not None:
      state["actor_critic"] = self.actor_critic.state_dict()
    if self.optimizer is not None:
      state["optimizer"] = self.optimizer.state_dict()
    return state

  def load(self, state: dict[str, Any]) -> None:
    self.device = state.get("device", self.device)
    self.learning_rate = state.get("learning_rate", self.learning_rate)
    if self.actor_critic is not None and "actor_critic" in state:
      self.actor_critic.load_state_dict(state["actor_critic"])
    if self.optimizer is not None and "optimizer" in state:
      self.optimizer.load_state_dict(state["optimizer"])

  def get_inference_policy(
    self,
  ) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    def _policy(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
      if self.actor_critic is None:
        return self.act(obs)
      with torch.no_grad():
        outputs = self.actor_critic.forward_train(obs)
      return {"actions": outputs["actions_mean"]}

    return _policy
