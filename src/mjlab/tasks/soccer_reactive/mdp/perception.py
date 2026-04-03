"""Virtual perception and shared runtime helpers for reactive soccer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.asset_zoo.robots import G1_COMP_BODY_JOINT_NAMES
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.soccer_reactive.mdp.landmarks import (
  LandmarkPerceptionConfig,
  VirtualLandmarkPerceptionChannel,
  build_field_landmark_map,
  build_landmark_observation,
  sample_virtual_landmark_detections,
)
from mjlab.tasks.soccer_reactive.mdp.localization_filter import ParticleFilterConfig
from mjlab.tasks.soccer_reactive.mdp.odometry import (
  ReactiveSoccerOdometryProxy,
  get_ground_truth_field_pose,
)
from mjlab.tasks.soccer.mdp import observations as soccer_obs

_PERCEPTION_RUNTIME_ATTR = "_soccer_reactive_perception_runtime"


def ball_noise_std(distance: torch.Tensor) -> torch.Tensor:
  """Distance-dependent ball observation noise from the paper."""
  return 0.124 * distance + 0.149


def detection_probability(
  distance: torch.Tensor,
  *,
  in_fov: torch.Tensor | None = None,
  near_probability: float = 0.9,
  guaranteed_range_m: float = 7.0,
  decay_distance_m: float = 3.0,
) -> torch.Tensor:
  """Detection probability used by the paper-style virtual perception model."""
  prob = torch.full_like(distance, fill_value=near_probability)
  far_mask = distance > guaranteed_range_m
  if torch.any(far_mask):
    prob[far_mask] = near_probability * torch.exp(
      -(distance[far_mask] - guaranteed_range_m) / max(decay_distance_m, 1.0e-6)
    )
  if in_fov is not None:
    prob = torch.where(in_fov, prob, torch.zeros_like(prob))
  return torch.clamp(prob, min=0.0, max=1.0)


@dataclass
class VirtualPerceptionState:
  """Legacy helper kept for unit tests and simple latency buffering."""

  latency_steps: int = 0
  update_interval_steps: int = 1
  _obs_history: list[torch.Tensor] = field(default_factory=list, init=False)
  _mask_history: list[torch.Tensor] = field(default_factory=list, init=False)
  _held_obs: torch.Tensor | None = field(default=None, init=False)
  _held_mask: torch.Tensor | None = field(default=None, init=False)
  _push_count: int = field(default=0, init=False)

  def push(self, obs: torch.Tensor, mask: torch.Tensor) -> None:
    self._obs_history.append(obs.clone())
    self._mask_history.append(mask.clone())
    self._push_count += 1

    history_index = max(0, len(self._obs_history) - 1 - self.latency_steps)
    should_update = self._held_obs is None or (
      (self._push_count - 1) % max(self.update_interval_steps, 1) == 0
    )
    if should_update:
      self._held_obs = self._obs_history[history_index]
      self._held_mask = self._mask_history[history_index]

  def read(self) -> tuple[torch.Tensor, torch.Tensor]:
    if self._held_obs is None or self._held_mask is None:
      raise ValueError("VirtualPerceptionState is empty; push observations first.")
    return self._held_obs.clone(), self._held_mask.clone()

  def reset(self) -> None:
    self._obs_history.clear()
    self._mask_history.clear()
    self._held_obs = None
    self._held_mask = None
    self._push_count = 0


class VirtualPerceptionChannel:
  """Asynchronous sample-and-hold perception stream with latency and age."""

  def __init__(
    self,
    *,
    num_envs: int,
    obs_dim: int,
    device: str,
    step_dt: float,
    mean_frequency_hz: float,
    frequency_std_hz: float,
    mean_latency_s: float,
    latency_std_s: float,
    max_latency_s: float | None = None,
  ) -> None:
    self.num_envs = num_envs
    self.obs_dim = obs_dim
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

    self._published_obs = torch.zeros((num_envs, obs_dim), device=device)
    self._published_mask = torch.zeros((num_envs, 1), device=device)
    self._observation_age = torch.zeros((num_envs, 1), device=device)
    self._time_until_next_capture = torch.zeros(num_envs, device=device)
    self._pending_obs = torch.zeros(
      (self.max_latency_steps + 1, num_envs, obs_dim), device=device
    )
    self._pending_mask = torch.zeros(
      (self.max_latency_steps + 1, num_envs, 1), device=device
    )
    self._pending_valid = torch.zeros(
      (self.max_latency_steps + 1, num_envs), device=device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    self._published_obs[env_ids] = 0.0
    self._published_mask[env_ids] = 0.0
    self._observation_age[env_ids] = 0.0
    self._time_until_next_capture[env_ids] = 0.0
    self._pending_obs[:, env_ids] = 0.0
    self._pending_mask[:, env_ids] = 0.0
    self._pending_valid[:, env_ids] = False

  def _sample_normal(self, shape: tuple[int, ...], mean: float, std: float) -> torch.Tensor:
    if std == 0.0:
      return torch.full(shape, fill_value=mean, device=self.device)
    return torch.randn(shape, device=self.device) * std + mean

  def _sample_capture_period(self, env_count: int) -> torch.Tensor:
    freq = self._sample_normal(
      (env_count,),
      mean=self.mean_frequency_hz,
      std=self.frequency_std_hz,
    ).clamp(min=1.0e-6)
    return 1.0 / freq

  def _sample_latency_steps(self, env_count: int) -> torch.Tensor:
    latency_s = self._sample_normal(
      (env_count,),
      mean=self.mean_latency_s,
      std=self.latency_std_s,
    ).clamp(min=0.0)
    return torch.round(latency_s / self.step_dt).long().clamp(
      min=0, max=self.max_latency_steps
    )

  def advance(
    self,
    *,
    true_obs: torch.Tensor,
    detection_mask: torch.Tensor,
    noise_std: torch.Tensor,
  ) -> None:
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

    self._observation_age += self.step_dt
    if torch.any(deliver_valid):
      self._published_obs[deliver_valid] = deliver_obs[deliver_valid]
      self._published_mask[deliver_valid] = deliver_mask[deliver_valid]
      self._observation_age[deliver_valid] = 0.0

    self._time_until_next_capture -= self.step_dt
    capture_due = self._time_until_next_capture <= 0.0
    if not torch.any(capture_due):
      return

    capture_ids = capture_due.nonzero(as_tuple=False).squeeze(-1)
    sampled_period = self._sample_capture_period(len(capture_ids))
    latency_steps = self._sample_latency_steps(len(capture_ids))
    capture_mask = detection_mask[capture_ids]
    sampled_obs = true_obs[capture_ids].clone()

    noise_scale = noise_std[capture_ids]
    if noise_scale.ndim == 1:
      noise_scale = noise_scale.unsqueeze(-1)
    sampled_obs = sampled_obs + torch.randn_like(sampled_obs) * noise_scale
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
        continue
      self._pending_obs[delay_steps, env_id] = obs
      self._pending_mask[delay_steps, env_id] = mask
      self._pending_valid[delay_steps, env_id] = True

  def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      self._published_obs.clone(),
      self._published_mask.clone(),
      self._observation_age.clone(),
    )


class ReactiveSoccerPerceptionRuntime:
  """Shared perception runtime so multiple observation groups stay consistent."""

  def __init__(self, env) -> None:
    self.device = env.device
    self._env = env
    self._last_step = -1
    self._joint_cfg = SceneEntityCfg("robot", joint_names=G1_COMP_BODY_JOINT_NAMES)
    self._joint_cfg.resolve(env.scene)

    perception_cfg = getattr(env.cfg, "perception", None)
    camera_hz = float(getattr(perception_cfg, "camera_hz", 30.0))
    detection_mean_hz = float(getattr(perception_cfg, "detection_mean_hz", 25.36))
    detection_std_hz = float(getattr(perception_cfg, "detection_std_hz", 1.06))
    latency_mean_s = float(getattr(perception_cfg, "latency_mean_s", 0.116))
    latency_std_s = float(getattr(perception_cfg, "latency_std_s", 0.018))

    self.ball_channel = VirtualPerceptionChannel(
      num_envs=env.num_envs,
      obs_dim=2,
      device=env.device,
      step_dt=env.step_dt,
      mean_frequency_hz=min(camera_hz, detection_mean_hz),
      frequency_std_hz=detection_std_hz,
      mean_latency_s=latency_mean_s,
      latency_std_s=latency_std_s,
    )

    self.landmark_map = build_field_landmark_map(device=env.device)
    self.landmark_perception_cfg = LandmarkPerceptionConfig(
      fov_half_angle_deg=float(getattr(perception_cfg, "landmark_fov_half_angle_deg", 90.0)),
      max_range_m=float(getattr(perception_cfg, "landmark_max_range_m", 12.0)),
      near_probability=float(getattr(perception_cfg, "landmark_near_probability", 0.9)),
      guaranteed_range_m=float(getattr(perception_cfg, "landmark_guaranteed_range_m", 7.0)),
      decay_distance_m=float(getattr(perception_cfg, "landmark_decay_distance_m", 3.0)),
      dropout_prob=float(getattr(perception_cfg, "landmark_dropout_prob", 0.05)),
      noise_std_base_m=float(getattr(perception_cfg, "landmark_noise_std_base_m", 0.03)),
      noise_std_scale_per_m=float(getattr(perception_cfg, "landmark_noise_std_scale_per_m", 0.01)),
    )
    self._pending_landmark_measurement = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.landmark_channel = VirtualLandmarkPerceptionChannel(
      num_envs=env.num_envs,
      num_landmarks=self.landmark_map.num_landmarks,
      device=env.device,
      step_dt=env.step_dt,
      mean_frequency_hz=float(getattr(perception_cfg, "landmark_mean_hz", detection_mean_hz)),
      frequency_std_hz=float(getattr(perception_cfg, "landmark_std_hz", detection_std_hz)),
      mean_latency_s=float(getattr(perception_cfg, "landmark_latency_mean_s", latency_mean_s)),
      latency_std_s=float(getattr(perception_cfg, "landmark_latency_std_s", latency_std_s)),
    )

    joint_dim = (
      len(self._joint_cfg.joint_ids)
      if isinstance(self._joint_cfg.joint_ids, list)
      else len(G1_COMP_BODY_JOINT_NAMES)
    )
    action_dim = int(getattr(env.action_manager, "total_action_dim", joint_dim))
    proprio_dim = 3 + 3 + joint_dim + joint_dim + action_dim
    self.odometry = ReactiveSoccerOdometryProxy(
      env=env,
      field_landmarks_f=self.landmark_map.positions_f,
      history_steps=int(getattr(perception_cfg, "odometry_history_steps", 50)),
      update_hz=float(getattr(perception_cfg, "odometry_update_hz", 20.0)),
      localization_mode=str(getattr(perception_cfg, "localization_mode", "particle_filter")),
      odometry_model_path=str(getattr(perception_cfg, "odometry_model_path", "") or ""),
      pf_cfg=ParticleFilterConfig(
        num_particles=int(getattr(perception_cfg, "pf_num_particles", 128)),
        resample_threshold=float(getattr(perception_cfg, "pf_resample_threshold", 0.5)),
        motion_noise_xy=float(getattr(perception_cfg, "pf_motion_noise_xy", 0.03)),
        motion_noise_yaw=float(getattr(perception_cfg, "pf_motion_noise_yaw", 0.05)),
        measurement_sigma=float(getattr(perception_cfg, "pf_measurement_sigma", 0.15)),
        initial_xy_std=float(getattr(perception_cfg, "pf_initial_xy_std", 0.02)),
        initial_yaw_std=float(getattr(perception_cfg, "pf_initial_yaw_std", 0.05)),
      ),
      expected_proprio_dim=proprio_dim,
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    self.ball_channel.reset(env_ids)
    self.landmark_channel.reset(env_ids)
    self.odometry.reset(env_ids)
    if env_ids is None:
      self._pending_landmark_measurement[:] = False
    else:
      self._pending_landmark_measurement[env_ids] = False
    self._last_step = -1

  def update(self, env) -> None:
    step = int(env.common_step_counter)
    if self._last_step == step:
      return

    true_pose_f = get_ground_truth_field_pose(env)

    ball_pos_b = soccer_obs.ball_pos_b(env)[:, :2]
    distance = torch.norm(ball_pos_b, dim=1)
    heading = torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])
    in_fov = (ball_pos_b[:, 0] > 0.0) & (torch.abs(heading) <= math.radians(75.0))
    detect_prob = detection_probability(distance, in_fov=in_fov)
    detect_mask = (
      torch.rand((env.num_envs,), device=env.device) < detect_prob
    ).float().unsqueeze(-1)
    self.ball_channel.advance(
      true_obs=ball_pos_b,
      detection_mask=detect_mask,
      noise_std=ball_noise_std(distance),
    )

    landmark_true_b, landmark_detect_mask, landmark_noise_std = sample_virtual_landmark_detections(
      pose_f=true_pose_f,
      landmark_positions_f=self.landmark_map.positions_f,
      cfg=self.landmark_perception_cfg,
    )
    self.landmark_channel.advance(
      true_obs=landmark_true_b,
      detection_mask=landmark_detect_mask,
      noise_std=landmark_noise_std,
    )
    landmark_pos_b, landmark_mask, landmark_age, landmark_publish = self.landmark_channel.read()
    self._pending_landmark_measurement |= landmark_publish
    landmark_observation = build_landmark_observation(
      landmark_pos_b,
      landmark_mask,
      landmark_age,
    )

    self.odometry.advance(
      true_pose_f=true_pose_f,
      landmark_observation=landmark_observation,
      landmark_measurement_fresh=self._pending_landmark_measurement,
      projected_gravity=envs_mdp.projected_gravity(env),
      base_ang_vel=envs_mdp.base_ang_vel(env),
      joint_pos=envs_mdp.joint_pos_rel(env, asset_cfg=self._joint_cfg),
      joint_vel=envs_mdp.joint_vel_rel(env, asset_cfg=self._joint_cfg),
      previous_action=envs_mdp.last_action(env),
    )

    self._pending_landmark_measurement[self.odometry.last_update_mask] = False

    odom_stats = self.odometry.latest_stats
    setattr(env, "_soccer_reactive_pose_hat_f", self.odometry.pose_hat_f.clone())
    setattr(env, "_soccer_reactive_goal_obs_age", self.odometry.goal_obs_age.clone())
    setattr(env, "_soccer_reactive_localization_success", odom_stats.localization_success.clone())
    setattr(env, "_soccer_reactive_visible_landmarks", odom_stats.visible_landmarks.clone())
    setattr(env, "_soccer_reactive_pose_error_xy", odom_stats.pose_error_xy.clone())
    setattr(env, "_soccer_reactive_pose_error_yaw", odom_stats.pose_error_yaw.clone())
    if odom_stats.particle_effective_n is not None:
      setattr(env, "_soccer_reactive_pf_effective_n", odom_stats.particle_effective_n.clone())
    self._last_step = step

  def read_ball(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return self.ball_channel.read()

  def read_goal(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return self.odometry.read()


def get_reactive_soccer_perception_runtime(env) -> ReactiveSoccerPerceptionRuntime:
  runtime = getattr(env, _PERCEPTION_RUNTIME_ATTR, None)
  if runtime is None:
    runtime = ReactiveSoccerPerceptionRuntime(env)
    setattr(env, _PERCEPTION_RUNTIME_ATTR, runtime)
  return runtime
