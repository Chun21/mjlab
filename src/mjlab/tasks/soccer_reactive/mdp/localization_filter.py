"""Localization utilities and particle filter for reactive soccer odometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import wrap_to_pi


@dataclass(frozen=True)
class ParticleFilterConfig:
  num_particles: int = 128
  resample_threshold: float = 0.5
  motion_noise_xy: float = 0.03
  motion_noise_yaw: float = 0.05
  measurement_sigma: float = 0.15
  initial_xy_std: float = 0.02
  initial_yaw_std: float = 0.05


@dataclass(frozen=True)
class LandmarkObservation:
  positions_b: torch.Tensor
  mask: torch.Tensor
  age: torch.Tensor

  def visible_counts(self) -> torch.Tensor:
    return self.mask.squeeze(-1).sum(dim=1)


def rotate_field_to_body(vec_f: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  cos_yaw = torch.cos(yaw)
  sin_yaw = torch.sin(yaw)
  x = vec_f[..., 0]
  y = vec_f[..., 1]
  x_b = cos_yaw * x + sin_yaw * y
  y_b = -sin_yaw * x + cos_yaw * y
  return torch.stack((x_b, y_b), dim=-1)


def rotate_body_to_field(vec_b: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  cos_yaw = torch.cos(yaw)
  sin_yaw = torch.sin(yaw)
  x = vec_b[..., 0]
  y = vec_b[..., 1]
  x_f = cos_yaw * x - sin_yaw * y
  y_f = sin_yaw * x + cos_yaw * y
  return torch.stack((x_f, y_f), dim=-1)


def integrate_delta_pose(
  pose_f: torch.Tensor,
  delta_pose_body: torch.Tensor,
) -> torch.Tensor:
  delta_xy_f = rotate_body_to_field(delta_pose_body[..., :2], pose_f[..., 2])
  next_pose = pose_f.clone()
  next_pose[..., :2] = next_pose[..., :2] + delta_xy_f
  next_pose[..., 2] = wrap_to_pi(next_pose[..., 2] + delta_pose_body[..., 2])
  return next_pose


def compute_body_frame_delta(
  prev_pose_f: torch.Tensor,
  next_pose_f: torch.Tensor,
) -> torch.Tensor:
  delta_xy_f = next_pose_f[..., :2] - prev_pose_f[..., :2]
  delta_xy_b = rotate_field_to_body(delta_xy_f, prev_pose_f[..., 2])
  delta_yaw = wrap_to_pi(next_pose_f[..., 2] - prev_pose_f[..., 2]).unsqueeze(-1)
  return torch.cat((delta_xy_b, delta_yaw), dim=-1)


def _broadcast_goal_vector(vector_f: torch.Tensor, pose_f: torch.Tensor) -> torch.Tensor:
  if vector_f.ndim == 1:
    return vector_f.unsqueeze(0).expand(pose_f.shape[0], -1)
  return vector_f


def goal_observation_from_pose(
  pose_f: torch.Tensor,
  goal_pos_f: torch.Tensor,
  goal_direction_f: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  goal_pos_f = _broadcast_goal_vector(goal_pos_f, pose_f)
  rel_goal_f = goal_pos_f - pose_f[..., :2]
  goal_pos_b = rotate_field_to_body(rel_goal_f, pose_f[..., 2])
  if goal_direction_f is None:
    goal_dir_b = goal_pos_b / torch.norm(goal_pos_b, dim=-1, keepdim=True).clamp(min=1.0e-6)
  else:
    goal_direction_f = _broadcast_goal_vector(goal_direction_f, pose_f)
    goal_dir_b = rotate_field_to_body(goal_direction_f, pose_f[..., 2])
    goal_dir_b = goal_dir_b / torch.norm(goal_dir_b, dim=-1, keepdim=True).clamp(min=1.0e-6)
  return goal_pos_b, goal_dir_b


class ParticleFilterLocalizer:
  """Minimal batched particle filter for field-local robot pose."""

  def __init__(
    self,
    *,
    num_envs: int,
    field_landmarks_f: torch.Tensor,
    device: str,
    cfg: ParticleFilterConfig,
  ) -> None:
    self.num_envs = num_envs
    self.field_landmarks_f = field_landmarks_f.to(device=device, dtype=torch.float32)
    self.device = device
    self.cfg = cfg
    self.num_particles = int(cfg.num_particles)
    self._particles = torch.zeros((num_envs, self.num_particles, 3), device=device)
    self._weights = torch.full(
      (num_envs, self.num_particles),
      1.0 / max(self.num_particles, 1),
      device=device,
    )

  @property
  def particles(self) -> torch.Tensor:
    return self._particles

  @property
  def weights(self) -> torch.Tensor:
    return self._weights

  def effective_sample_size(self) -> torch.Tensor:
    return 1.0 / self._weights.pow(2).sum(dim=-1).clamp(min=1.0e-9)

  def reset(self, pose_f: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    if len(env_ids) == 0:
      return
    base_pose = pose_f[env_ids].unsqueeze(1).repeat(1, self.num_particles, 1)
    if self.cfg.initial_xy_std > 0.0:
      base_pose[..., :2] += torch.randn_like(base_pose[..., :2]) * self.cfg.initial_xy_std
    if self.cfg.initial_yaw_std > 0.0:
      base_pose[..., 2] = wrap_to_pi(
        base_pose[..., 2] + torch.randn_like(base_pose[..., 2]) * self.cfg.initial_yaw_std
      )
    self._particles[env_ids] = base_pose
    self._weights[env_ids] = 1.0 / max(self.num_particles, 1)

  def predict(
    self,
    env_ids: torch.Tensor,
    delta_pose_body: torch.Tensor,
  ) -> None:
    if len(env_ids) == 0:
      return
    particles = self._particles[env_ids]
    delta = delta_pose_body.unsqueeze(1).repeat(1, self.num_particles, 1)
    if self.cfg.motion_noise_xy > 0.0:
      delta[..., :2] = delta[..., :2] + torch.randn_like(delta[..., :2]) * self.cfg.motion_noise_xy
    if self.cfg.motion_noise_yaw > 0.0:
      delta[..., 2] = delta[..., 2] + torch.randn_like(delta[..., 2]) * self.cfg.motion_noise_yaw
    self._particles[env_ids] = integrate_delta_pose(particles, delta)

  def update(
    self,
    env_ids: torch.Tensor,
    landmark_observation: LandmarkObservation,
  ) -> torch.Tensor:
    if len(env_ids) == 0:
      return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    sigma = max(float(self.cfg.measurement_sigma), 1.0e-6)
    positions_b = landmark_observation.positions_b[env_ids]
    mask = landmark_observation.mask[env_ids].squeeze(-1).bool()
    particles = self._particles[env_ids]
    weights = self._weights[env_ids]

    rel_landmarks_f = self.field_landmarks_f.unsqueeze(0).unsqueeze(0) - particles[:, :, None, :2]
    predicted_b = rotate_field_to_body(rel_landmarks_f, particles[:, :, None, 2])
    residual = predicted_b - positions_b[:, None, :, :]
    squared_error = residual.pow(2).sum(dim=-1)
    masked_error = torch.where(mask[:, None, :], squared_error, torch.zeros_like(squared_error))
    visible_counts = mask.sum(dim=-1)
    has_measurement = visible_counts > 0
    if torch.any(has_measurement):
      log_likelihood = -0.5 * masked_error.sum(dim=-1) / (sigma * sigma)
      max_log = log_likelihood.max(dim=-1, keepdim=True).values
      unnormalized = torch.exp(log_likelihood - max_log) * weights
      normalized = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp(min=1.0e-9)
      weights = torch.where(has_measurement.unsqueeze(-1), normalized, weights)
      self._weights[env_ids] = weights
      self._maybe_resample(env_ids[has_measurement])
    success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    success[env_ids] = has_measurement
    return success

  def estimate(self) -> torch.Tensor:
    weight = self._weights.unsqueeze(-1)
    xy = (self._particles[..., :2] * weight).sum(dim=1)
    yaw_sin = (torch.sin(self._particles[..., 2]) * self._weights).sum(dim=1)
    yaw_cos = (torch.cos(self._particles[..., 2]) * self._weights).sum(dim=1)
    yaw = torch.atan2(yaw_sin, yaw_cos)
    return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)

  def _maybe_resample(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    effective_n = self.effective_sample_size()[env_ids]
    resample_mask = effective_n < (self.cfg.resample_threshold * self.num_particles)
    if not torch.any(resample_mask):
      return
    resample_ids = env_ids[resample_mask]
    resample_weights = self._weights[resample_ids]
    indices = torch.multinomial(resample_weights, num_samples=self.num_particles, replacement=True)
    env_offsets = torch.arange(len(resample_ids), device=self.device).unsqueeze(-1)
    self._particles[resample_ids] = self._particles[resample_ids][env_offsets, indices]
    self._weights[resample_ids] = 1.0 / max(self.num_particles, 1)
