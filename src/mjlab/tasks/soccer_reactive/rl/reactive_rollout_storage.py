"""Minimal dual-critic rollout storage for reactive soccer."""

from __future__ import annotations

import torch


class ReactiveRolloutStorage:
  """Track dual-critic values, returns and reconstruction targets."""

  @staticmethod
  def _sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

  def __init__(
    self,
    num_envs: int,
    num_steps: int,
    actor_obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    device: str,
    critic_obs_shape: tuple[int, ...] | None = None,
    reconstruction_shape: tuple[int, ...] = (18,),
    actor_history_shape: tuple[int, ...] | None = None,
    privileged_shape: tuple[int, ...] | None = None,
  ) -> None:
    self.num_envs = num_envs
    self.num_steps = num_steps
    self.actor_obs_shape = actor_obs_shape
    self.critic_obs_shape = actor_obs_shape if critic_obs_shape is None else critic_obs_shape
    self.action_shape = action_shape
    self.device = device
    self.actor_history_shape = actor_history_shape
    self.privileged_shape = privileged_shape
    self.step = 0

    self.goal_values = torch.zeros((num_steps, num_envs, 1), device=device)
    self.aux_values = torch.zeros((num_steps, num_envs, 1), device=device)
    self.goal_returns = torch.zeros((num_steps, num_envs, 1), device=device)
    self.aux_returns = torch.zeros((num_steps, num_envs, 1), device=device)
    self.goal_advantages = torch.zeros((num_steps, num_envs, 1), device=device)
    self.aux_advantages = torch.zeros((num_steps, num_envs, 1), device=device)
    self.goal_rewards = torch.zeros((num_steps, num_envs, 1), device=device)
    self.aux_rewards = torch.zeros((num_steps, num_envs, 1), device=device)
    self.dones = torch.zeros((num_steps, num_envs, 1), device=device)
    self.actions = torch.zeros((num_steps, num_envs, *action_shape), device=device)
    self.old_action_mean = torch.zeros((num_steps, num_envs, *action_shape), device=device)
    self.old_log_probs = torch.zeros((num_steps, num_envs, 1), device=device)
    self.actor_current = torch.zeros((num_steps, num_envs, *actor_obs_shape), device=device)
    self.critic_current = torch.zeros(
      (num_steps, num_envs, *self.critic_obs_shape), device=device
    )
    self.reconstruction_targets = torch.zeros(
      (num_steps, num_envs, *reconstruction_shape),
      device=device,
    )
    if actor_history_shape is not None:
      self.actor_history = torch.zeros(
        (num_steps, num_envs, *actor_history_shape),
        device=device,
      )
    else:
      self.actor_history = None
    if privileged_shape is not None:
      self.critic_privileged = torch.zeros(
        (num_steps, num_envs, *privileged_shape),
        device=device,
      )
    else:
      self.critic_privileged = None

  def reset(self) -> None:
    self.step = 0

  def add(
    self,
    *,
    actor_current: torch.Tensor,
    critic_current: torch.Tensor | None,
    actor_history: torch.Tensor | None,
    critic_privileged: torch.Tensor | None,
    actions: torch.Tensor,
    old_action_mean: torch.Tensor,
    old_log_probs: torch.Tensor,
    goal_values: torch.Tensor,
    aux_values: torch.Tensor,
    goal_rewards: torch.Tensor,
    aux_rewards: torch.Tensor,
    dones: torch.Tensor,
    reconstruction_targets: torch.Tensor,
  ) -> None:
    if self.step >= self.num_steps:
      raise RuntimeError("ReactiveRolloutStorage is full")

    idx = self.step
    self.actor_current[idx].copy_(self._sanitize_tensor(actor_current))
    if critic_current is None:
      critic_current = actor_current
    self.critic_current[idx].copy_(self._sanitize_tensor(critic_current))
    if self.actor_history is not None and actor_history is not None:
      self.actor_history[idx].copy_(self._sanitize_tensor(actor_history))
    if self.critic_privileged is not None and critic_privileged is not None:
      self.critic_privileged[idx].copy_(self._sanitize_tensor(critic_privileged))
    self.actions[idx].copy_(self._sanitize_tensor(actions))
    self.old_action_mean[idx].copy_(self._sanitize_tensor(old_action_mean))
    self.old_log_probs[idx].copy_(self._sanitize_tensor(old_log_probs))
    self.goal_values[idx].copy_(self._sanitize_tensor(goal_values))
    self.aux_values[idx].copy_(self._sanitize_tensor(aux_values))
    self.goal_rewards[idx].copy_(self._sanitize_tensor(goal_rewards))
    self.aux_rewards[idx].copy_(self._sanitize_tensor(aux_rewards))
    self.dones[idx].copy_(self._sanitize_tensor(dones))
    self.reconstruction_targets[idx].copy_(self._sanitize_tensor(reconstruction_targets))
    self.step += 1

  def compute_returns(
    self,
    next_goal_value: torch.Tensor,
    next_aux_value: torch.Tensor,
    gamma: float,
    lam: float,
  ) -> None:
    steps = self.step
    goal_adv = torch.zeros((self.num_envs, 1), device=self.device)
    aux_adv = torch.zeros((self.num_envs, 1), device=self.device)
    next_goal_value = self._sanitize_tensor(next_goal_value)
    next_aux_value = self._sanitize_tensor(next_aux_value)
    for idx in reversed(range(steps)):
      not_done = 1.0 - self._sanitize_tensor(self.dones[idx])
      next_gv = next_goal_value if idx == steps - 1 else self.goal_values[idx + 1]
      next_av = next_aux_value if idx == steps - 1 else self.aux_values[idx + 1]

      goal_delta = self.goal_rewards[idx] + gamma * next_gv * not_done - self.goal_values[idx]
      aux_delta = self.aux_rewards[idx] + gamma * next_av * not_done - self.aux_values[idx]
      goal_adv = self._sanitize_tensor(goal_delta + gamma * lam * not_done * goal_adv)
      aux_adv = self._sanitize_tensor(aux_delta + gamma * lam * not_done * aux_adv)
      self.goal_advantages[idx].copy_(goal_adv)
      self.aux_advantages[idx].copy_(aux_adv)
      self.goal_returns[idx].copy_(self._sanitize_tensor(goal_adv + self.goal_values[idx]))
      self.aux_returns[idx].copy_(self._sanitize_tensor(aux_adv + self.aux_values[idx]))

  def flattened_batch(self) -> dict[str, torch.Tensor]:
    steps = self.step
    batch: dict[str, torch.Tensor] = {
      "actor_current": self.actor_current[:steps].reshape(-1, *self.actor_obs_shape),
      "critic_current": self.critic_current[:steps].reshape(-1, *self.critic_obs_shape),
      "actions": self.actions[:steps].reshape(-1, *self.action_shape),
      "old_action_mean": self.old_action_mean[:steps].reshape(-1, *self.action_shape),
      "old_log_probs": self.old_log_probs[:steps].reshape(-1, 1),
      "goal_values": self.goal_values[:steps].reshape(-1, 1),
      "aux_values": self.aux_values[:steps].reshape(-1, 1),
      "goal_returns": self.goal_returns[:steps].reshape(-1, 1),
      "aux_returns": self.aux_returns[:steps].reshape(-1, 1),
      "goal_advantages": self.goal_advantages[:steps].reshape(-1, 1),
      "aux_advantages": self.aux_advantages[:steps].reshape(-1, 1),
      "reconstruction_targets": self.reconstruction_targets[:steps].reshape(
        -1, *self.reconstruction_targets.shape[2:]
      ),
    }
    if self.actor_history is not None:
      batch["actor_history"] = self.actor_history[:steps].reshape(
        -1, *self.actor_history.shape[2:]
      )
    if self.critic_privileged is not None:
      batch["critic_privileged"] = self.critic_privileged[:steps].reshape(
        -1, *self.critic_privileged.shape[2:]
      )
    return batch
