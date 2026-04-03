"""Field-landmark map and virtual perception helpers for reactive soccer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.tasks.soccer.field_specs import M_FIELD, SoccerFieldConfig
from mjlab.tasks.soccer_reactive.mdp.localization_filter import LandmarkObservation, rotate_field_to_body


def _detection_probability(
  distance: torch.Tensor,
  *,
  in_fov: torch.Tensor | None = None,
  near_probability: float = 0.9,
  guaranteed_range_m: float = 7.0,
  decay_distance_m: float = 3.0,
) -> torch.Tensor:
  prob = torch.full_like(distance, fill_value=near_probability)
  far_mask = distance > guaranteed_range_m
  if torch.any(far_mask):
    prob[far_mask] = near_probability * torch.exp(
      -(distance[far_mask] - guaranteed_range_m) / max(decay_distance_m, 1.0e-6)
    )
  if in_fov is not None:
    prob = torch.where(in_fov, prob, torch.zeros_like(prob))
  return torch.clamp(prob, min=0.0, max=1.0)


@dataclass(frozen=True)
class FieldLandmarkSpec:
  """A single landmark on the field map."""

  name: str
  kind: str
  position_f: tuple[float, float]


@dataclass(frozen=True)
class FieldLandmarkMap:
  """Ordered field landmark map used by virtual perception and PF localization."""

  specs: tuple[FieldLandmarkSpec, ...]
  positions_f: torch.Tensor

  @property
  def num_landmarks(self) -> int:
    return len(self.specs)

  @property
  def names(self) -> tuple[str, ...]:
    return tuple(spec.name for spec in self.specs)


@dataclass(frozen=True)
class LandmarkPerceptionConfig:
  """Virtual landmark-perception parameters."""

  fov_half_angle_deg: float = 90.0
  max_range_m: float = 12.0
  near_probability: float = 0.9
  guaranteed_range_m: float = 7.0
  decay_distance_m: float = 3.0
  dropout_prob: float = 0.05
  noise_std_base_m: float = 0.03
  noise_std_scale_per_m: float = 0.01


class VirtualLandmarkPerceptionChannel:
  """Asynchronous sample-and-hold perception stream for field landmarks."""

  def __init__(
    self,
    *,
    num_envs: int,
    num_landmarks: int,
    device: str,
    step_dt: float,
    mean_frequency_hz: float,
    frequency_std_hz: float,
    mean_latency_s: float,
    latency_std_s: float,
    max_latency_s: float | None = None,
  ) -> None:
    self.num_envs = num_envs
    self.num_landmarks = num_landmarks
    self.device = device
    self.step_dt = float(step_dt)
    self.mean_frequency_hz = float(mean_frequency_hz)
    self.frequency_std_hz = float(frequency_std_hz)
    self.mean_latency_s = float(mean_latency_s)
    self.latency_std_s = float(latency_std_s)
    latency_ceiling = (
      max_latency_s
      if max_latency_s is not None
      else max(self.mean_latency_s + 4.0 * self.latency_std_s, self.step_dt)
    )
    self.max_latency_steps = max(1, int(math.ceil(latency_ceiling / self.step_dt)))

    self._published_obs = torch.zeros((num_envs, num_landmarks, 2), device=device)
    self._published_mask = torch.zeros((num_envs, num_landmarks, 1), device=device)
    self._observation_age = torch.zeros((num_envs, 1), device=device)
    self._time_until_next_capture = torch.zeros(num_envs, device=device)
    self._publish_event = torch.zeros(num_envs, device=device, dtype=torch.bool)
    self._pending_obs = torch.zeros(
      (self.max_latency_steps + 1, num_envs, num_landmarks, 2),
      device=device,
    )
    self._pending_mask = torch.zeros(
      (self.max_latency_steps + 1, num_envs, num_landmarks, 1),
      device=device,
    )
    self._pending_valid = torch.zeros(
      (self.max_latency_steps + 1, num_envs),
      device=device,
      dtype=torch.bool,
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    self._published_obs[env_ids] = 0.0
    self._published_mask[env_ids] = 0.0
    self._observation_age[env_ids] = 0.0
    self._time_until_next_capture[env_ids] = 0.0
    self._publish_event[env_ids] = False
    self._pending_obs[:, env_ids] = 0.0
    self._pending_mask[:, env_ids] = 0.0
    self._pending_valid[:, env_ids] = False

  def _sample_normal(self, shape: tuple[int, ...], mean: float, std: float) -> torch.Tensor:
    if std == 0.0:
      return torch.full(shape, fill_value=mean, device=self.device)
    return torch.randn(shape, device=self.device) * std + mean

  def _sample_capture_period(self, env_count: int) -> torch.Tensor:
    frequency = self._sample_normal(
      (env_count,),
      mean=self.mean_frequency_hz,
      std=self.frequency_std_hz,
    ).clamp(min=1.0e-6)
    return 1.0 / frequency

  def _sample_latency_steps(self, env_count: int) -> torch.Tensor:
    latency_s = self._sample_normal(
      (env_count,),
      mean=self.mean_latency_s,
      std=self.latency_std_s,
    ).clamp(min=0.0)
    return torch.round(latency_s / self.step_dt).long().clamp(
      min=0,
      max=self.max_latency_steps,
    )

  def advance(
    self,
    *,
    true_obs: torch.Tensor,
    detection_mask: torch.Tensor,
    noise_std: torch.Tensor,
  ) -> torch.Tensor:
    self._pending_obs[:-1] = self._pending_obs[1:].clone()
    self._pending_mask[:-1] = self._pending_mask[1:].clone()
    self._pending_valid[:-1] = self._pending_valid[1:].clone()
    deliver_obs = self._pending_obs[0].clone()
    deliver_mask = self._pending_mask[0].clone()
    deliver_valid = self._pending_valid[0].clone()
    self._pending_obs[0] = 0.0
    self._pending_mask[0] = 0.0
    self._pending_valid[0] = False
    self._pending_obs[-1] = 0.0
    self._pending_mask[-1] = 0.0
    self._pending_valid[-1] = False

    self._publish_event = deliver_valid.clone()
    self._observation_age += self.step_dt
    if torch.any(deliver_valid):
      self._published_obs[deliver_valid] = deliver_obs[deliver_valid]
      self._published_mask[deliver_valid] = deliver_mask[deliver_valid]
      self._observation_age[deliver_valid] = 0.0

    self._time_until_next_capture -= self.step_dt
    capture_due = self._time_until_next_capture <= 0.0
    if not torch.any(capture_due):
      return self._publish_event.clone()

    capture_ids = capture_due.nonzero(as_tuple=False).squeeze(-1)
    sampled_period = self._sample_capture_period(len(capture_ids))
    latency_steps = self._sample_latency_steps(len(capture_ids))
    capture_mask = detection_mask[capture_ids].clone()
    if capture_mask.ndim == 2:
      capture_mask = capture_mask.unsqueeze(-1)
    sampled_obs = true_obs[capture_ids].clone()
    sampled_noise = noise_std[capture_ids]
    if sampled_noise.ndim == 2:
      sampled_noise = sampled_noise.unsqueeze(-1)
    sampled_obs = sampled_obs + torch.randn_like(sampled_obs) * sampled_noise
    sampled_obs = sampled_obs * capture_mask

    self._time_until_next_capture[capture_ids] = sampled_period
    for local_idx, env_id in enumerate(capture_ids.tolist()):
      delay_steps = int(latency_steps[local_idx].item())
      obs = sampled_obs[local_idx]
      mask = capture_mask[local_idx]
      if delay_steps == 0:
        self._published_obs[env_id] = obs
        self._published_mask[env_id] = mask
        self._observation_age[env_id] = 0.0
        self._publish_event[env_id] = True
        continue
      self._pending_obs[delay_steps, env_id] = obs
      self._pending_mask[delay_steps, env_id] = mask
      self._pending_valid[delay_steps, env_id] = True

    return self._publish_event.clone()

  def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      self._published_obs.clone(),
      self._published_mask.clone(),
      self._observation_age.clone(),
      self._publish_event.clone(),
    )


def build_field_landmark_specs(field: SoccerFieldConfig = M_FIELD) -> tuple[FieldLandmarkSpec, ...]:
  """Build the paper-style landmark set: goalposts + T/X/L intersections."""

  half_length = field.field_length * 0.5
  half_width = field.field_width * 0.5
  half_goal_width = field.goal_width * 0.5
  half_goal_area_width = field.goal_area_width * 0.5
  half_penalty_area_width = field.penalty_area_width * 0.5
  center_circle_radius = field.center_circle_dia * 0.5

  specs: list[FieldLandmarkSpec] = []

  def add(name: str, kind: str, x: float, y: float) -> None:
    specs.append(FieldLandmarkSpec(name=name, kind=kind, position_f=(float(x), float(y))))

  # Goalposts.
  add("goalpost_left_top", "goalpost", -half_length, half_goal_width)
  add("goalpost_left_bottom", "goalpost", -half_length, -half_goal_width)
  add("goalpost_right_top", "goalpost", half_length, half_goal_width)
  add("goalpost_right_bottom", "goalpost", half_length, -half_goal_width)

  # L intersections: field outer corners + area front corners.
  for sign_x, side in ((-1.0, "left"), (1.0, "right")):
    x_goal_front = sign_x * (half_length - field.goal_area_length)
    x_penalty_front = sign_x * (half_length - field.penalty_area_length)
    x_boundary = sign_x * half_length
    add(f"corner_{side}_top", "L", x_boundary, half_width)
    add(f"corner_{side}_bottom", "L", x_boundary, -half_width)
    add(f"goal_area_{side}_front_top", "L", x_goal_front, half_goal_area_width)
    add(f"goal_area_{side}_front_bottom", "L", x_goal_front, -half_goal_area_width)
    add(f"penalty_area_{side}_front_top", "L", x_penalty_front, half_penalty_area_width)
    add(f"penalty_area_{side}_front_bottom", "L", x_penalty_front, -half_penalty_area_width)

  # T intersections: sidelines, goal-line/area intersections.
  add("center_line_top", "T", 0.0, half_width)
  add("center_line_bottom", "T", 0.0, -half_width)
  for sign_x, side in ((-1.0, "left"), (1.0, "right")):
    x_goal_line = sign_x * half_length
    add(f"goal_area_{side}_goalline_top", "T", x_goal_line, half_goal_area_width)
    add(f"goal_area_{side}_goalline_bottom", "T", x_goal_line, -half_goal_area_width)
    add(f"penalty_area_{side}_goalline_top", "T", x_goal_line, half_penalty_area_width)
    add(f"penalty_area_{side}_goalline_bottom", "T", x_goal_line, -half_penalty_area_width)

  # X intersections: center circle crossing the center line.
  add("center_circle_top", "X", 0.0, center_circle_radius)
  add("center_circle_bottom", "X", 0.0, -center_circle_radius)

  return tuple(specs)


def build_field_landmark_map(
  *,
  field: SoccerFieldConfig = M_FIELD,
  device: str | torch.device = "cpu",
) -> FieldLandmarkMap:
  specs = build_field_landmark_specs(field=field)
  positions_f = torch.tensor(
    [spec.position_f for spec in specs],
    dtype=torch.float32,
    device=device,
  )
  return FieldLandmarkMap(specs=specs, positions_f=positions_f)


def landmark_positions_in_body_frame(
  pose_f: torch.Tensor,
  landmark_positions_f: torch.Tensor,
) -> torch.Tensor:
  """Project field landmarks into the robot body frame."""

  rel_landmarks_f = landmark_positions_f.unsqueeze(0) - pose_f[:, None, :2]
  return rotate_field_to_body(rel_landmarks_f, pose_f[:, None, 2])


def sample_virtual_landmark_detections(
  *,
  pose_f: torch.Tensor,
  landmark_positions_f: torch.Tensor,
  cfg: LandmarkPerceptionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Generate paper-style virtual landmark detections in the body frame."""

  true_positions_b = landmark_positions_in_body_frame(pose_f, landmark_positions_f)
  distance = torch.norm(true_positions_b, dim=-1)
  heading = torch.atan2(true_positions_b[..., 1], true_positions_b[..., 0])
  in_fov = (
    (true_positions_b[..., 0] > 0.0)
    & (distance <= float(cfg.max_range_m))
    & (torch.abs(heading) <= math.radians(float(cfg.fov_half_angle_deg)))
  )
  detect_prob = _detection_probability(
    distance,
    in_fov=in_fov,
    near_probability=float(cfg.near_probability),
    guaranteed_range_m=float(cfg.guaranteed_range_m),
    decay_distance_m=float(cfg.decay_distance_m),
  )
  detect_prob = detect_prob * max(0.0, 1.0 - float(cfg.dropout_prob))
  detect_mask = (
    torch.rand_like(detect_prob) < detect_prob.clamp(min=0.0, max=1.0)
  ).float().unsqueeze(-1)
  noise_std = (
    float(cfg.noise_std_base_m) + float(cfg.noise_std_scale_per_m) * distance
  ).unsqueeze(-1)
  return true_positions_b, detect_mask, noise_std


def build_landmark_observation(
  positions_b: torch.Tensor,
  mask: torch.Tensor,
  age: torch.Tensor,
) -> LandmarkObservation:
  return LandmarkObservation(
    positions_b=positions_b,
    mask=mask,
    age=age,
  )
