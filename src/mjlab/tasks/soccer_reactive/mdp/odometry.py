"""Odometry-style goal stream proxy for reactive soccer."""

from __future__ import annotations

import torch

from mjlab.utils.buffers.circular_buffer import CircularBuffer


def build_odometry_proprio_vector(
  *,
  projected_gravity: torch.Tensor,
  base_ang_vel: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  previous_action: torch.Tensor,
) -> torch.Tensor:
  """Build the 1-second proprioceptive input used by the odometry proxy."""
  return torch.cat(
    (
      projected_gravity,
      base_ang_vel,
      joint_pos,
      joint_vel,
      previous_action,
    ),
    dim=-1,
  )


class ReactiveSoccerOdometryProxy:
  """Stateful odometry proxy with 1-second history and sample-and-hold output."""

  def __init__(
    self,
    *,
    num_envs: int,
    device: str,
    step_dt: float,
    history_steps: int = 50,
    update_hz: float = 20.0,
    teacher_correction_gain: float = 1.0,
    max_teacher_correction: float = 0.35,
  ) -> None:
    self.num_envs = num_envs
    self.device = device
    self.step_dt = float(step_dt)
    self.history_steps = int(history_steps)
    self.update_hz = float(update_hz)
    self.teacher_correction_gain = float(teacher_correction_gain)
    self.max_teacher_correction = float(max_teacher_correction)
    self.update_period_s = 1.0 / max(self.update_hz, 1.0e-6)

    self.proprio_history = CircularBuffer(
      max_len=self.history_steps,
      batch_size=num_envs,
      device=device,
    )
    self.goal_pos_b = torch.zeros((num_envs, 2), device=device)
    self.goal_dir_b = torch.zeros((num_envs, 2), device=device)
    self.goal_dir_b[:, 0] = 1.0
    self.observation_age = torch.zeros((num_envs, 1), device=device)
    self._time_since_update = torch.zeros(num_envs, device=device)
    self._is_bootstrapped = torch.zeros(num_envs, device=device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    self.goal_pos_b[env_ids] = 0.0
    self.goal_dir_b[env_ids] = 0.0
    self.goal_dir_b[env_ids, 0] = 1.0
    self.observation_age[env_ids] = 0.0
    self._time_since_update[env_ids] = 0.0
    self._is_bootstrapped[env_ids] = False
    if self.proprio_history.is_initialized:
      self.proprio_history.reset(batch_ids=env_ids)

  def advance(
    self,
    *,
    true_goal_pos_b: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    previous_action: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proprio_vec = build_odometry_proprio_vector(
      projected_gravity=projected_gravity,
      base_ang_vel=base_ang_vel,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      previous_action=previous_action,
    )
    self.proprio_history.append(proprio_vec)
    self.observation_age += self.step_dt
    self._time_since_update += self.step_dt

    bootstrap_ids = (~self._is_bootstrapped).nonzero(as_tuple=False).squeeze(-1)
    if len(bootstrap_ids) > 0:
      self.goal_pos_b[bootstrap_ids] = true_goal_pos_b[bootstrap_ids]
      self.goal_dir_b[bootstrap_ids] = self._normalize_goal_dir(true_goal_pos_b[bootstrap_ids])
      self.observation_age[bootstrap_ids] = 0.0
      self._time_since_update[bootstrap_ids] = 0.0
      self._is_bootstrapped[bootstrap_ids] = True

    update_ids = (self._time_since_update >= (self.update_period_s - 1.0e-6)).nonzero(
      as_tuple=False
    ).squeeze(-1)
    if len(update_ids) > 0:
      history = self.proprio_history.buffer[update_ids]
      ang_vel_hist = history[:, :, 3:6]
      yaw_rate = ang_vel_hist[:, :, 2].mean(dim=1, keepdim=True)
      predicted_goal = self._rotate_goal(
        self.goal_pos_b[update_ids],
        yaw_delta=-yaw_rate * self.update_period_s,
      )
      teacher_delta = true_goal_pos_b[update_ids] - predicted_goal
      teacher_delta_norm = torch.norm(teacher_delta, dim=1, keepdim=True).clamp(
        min=1.0e-6
      )
      bounded_teacher_delta = teacher_delta * (
        torch.clamp(teacher_delta_norm, max=self.max_teacher_correction)
        / teacher_delta_norm
      )
      corrected_goal = predicted_goal + self.teacher_correction_gain * bounded_teacher_delta
      corrected_goal[:, 1:2] = corrected_goal[:, 1:2] + 0.02 * torch.tanh(yaw_rate)
      self.goal_pos_b[update_ids] = corrected_goal
      self.goal_dir_b[update_ids] = self._normalize_goal_dir(corrected_goal)
      self.observation_age[update_ids] = 0.0
      self._time_since_update[update_ids] = (
        self._time_since_update[update_ids] - self.update_period_s
      ).clamp(min=0.0)

    return self.read()

  def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      self.goal_pos_b.clone(),
      self.goal_dir_b.clone(),
      self.observation_age.clone(),
    )

  @staticmethod
  def _normalize_goal_dir(goal_pos_b: torch.Tensor) -> torch.Tensor:
    return goal_pos_b / torch.norm(goal_pos_b, dim=-1, keepdim=True).clamp(min=1.0e-6)

  @staticmethod
  def _rotate_goal(goal_pos_b: torch.Tensor, yaw_delta: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw_delta)
    sin_yaw = torch.sin(yaw_delta)
    x = goal_pos_b[:, 0:1]
    y = goal_pos_b[:, 1:2]
    rotated_x = cos_yaw * x - sin_yaw * y
    rotated_y = sin_yaw * x + cos_yaw * y
    return torch.cat((rotated_x, rotated_y), dim=-1)
