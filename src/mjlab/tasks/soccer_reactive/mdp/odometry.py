"""Paper-style learned odometry runtime for reactive soccer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.entity import Entity
from mjlab.tasks.soccer.field_specs import M_FIELD
from mjlab.tasks.soccer_reactive.mdp.localization_filter import (
  LandmarkObservation,
  ParticleFilterConfig,
  ParticleFilterLocalizer,
  compute_body_frame_delta,
  goal_observation_from_pose,
  integrate_delta_pose,
)
from mjlab.tasks.soccer_reactive.models.odometry_mlp import (
  ReactiveSoccerOdometryMLP,
  ReactiveSoccerOdometryModelMetadata,
  load_reactive_soccer_odometry_checkpoint,
)
from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi


_GOAL_POS_F = torch.tensor((M_FIELD.field_length * 0.5, 0.0), dtype=torch.float32)
_GOAL_DIR_F = torch.tensor((1.0, 0.0), dtype=torch.float32)
_VALID_LOCALIZATION_MODES = frozenset({"ground_truth", "learned_only", "particle_filter"})


@dataclass(frozen=True)
class ReactiveSoccerOdometryStats:
  """Runtime diagnostics for localization health monitoring."""

  localization_success: torch.Tensor
  visible_landmarks: torch.Tensor
  pose_error_xy: torch.Tensor
  pose_error_yaw: torch.Tensor
  particle_effective_n: torch.Tensor | None = None


def build_odometry_proprio_vector(
  *,
  projected_gravity: torch.Tensor,
  base_ang_vel: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  previous_action: torch.Tensor,
) -> torch.Tensor:
  """Build the 1-second proprioceptive input used by the paper odometry."""
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


def get_ground_truth_field_pose(env) -> torch.Tensor:
  """Return the robot pose in the field-local frame ``[x, y, yaw]``."""

  robot: Entity = env.scene["robot"]
  root_pos_w = robot.data.root_link_pos_w[:, :2]
  env_origin_xy = env.scene.env_origins[:, :2]
  _, _, yaw_w = euler_xyz_from_quat(robot.data.root_link_quat_w)
  pose_f = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
  pose_f[:, :2] = root_pos_w - env_origin_xy
  pose_f[:, 2] = wrap_to_pi(yaw_w)
  return pose_f


class ReactiveSoccerLearnedOdometryRuntime:
  """Field-local learned odometry with optional particle-filter correction."""

  def __init__(
    self,
    *,
    env,
    field_landmarks_f: torch.Tensor,
    history_steps: int = 50,
    update_hz: float = 20.0,
    localization_mode: str = "particle_filter",
    odometry_model_path: str = "",
    pf_cfg: ParticleFilterConfig | None = None,
    expected_proprio_dim: int | None = None,
    goal_pos_f: torch.Tensor | None = None,
    goal_direction_f: torch.Tensor | None = None,
  ) -> None:
    self.env = env
    self.num_envs = env.num_envs
    self.device = env.device
    self.step_dt = float(env.step_dt)
    self.history_steps = int(history_steps)
    self.update_hz = float(update_hz)
    self.update_period_s = 1.0 / max(self.update_hz, 1.0e-6)
    self.localization_mode = str(localization_mode)
    if self.localization_mode not in _VALID_LOCALIZATION_MODES:
      raise ValueError(
        "Unsupported localization_mode "
        f"{self.localization_mode!r}; expected one of {sorted(_VALID_LOCALIZATION_MODES)!r}."
      )

    self.goal_pos_f = (goal_pos_f if goal_pos_f is not None else _GOAL_POS_F).to(
      device=self.device,
      dtype=torch.float32,
    )
    self.goal_direction_f = (
      goal_direction_f if goal_direction_f is not None else _GOAL_DIR_F
    ).to(device=self.device, dtype=torch.float32)
    self.field_landmarks_f = field_landmarks_f.to(device=self.device, dtype=torch.float32)

    self.pose_hat_f = torch.zeros((self.num_envs, 3), device=self.device)
    self.prev_delta_pose = torch.zeros((self.num_envs, 3), device=self.device)
    self.goal_pos_b = torch.zeros((self.num_envs, 2), device=self.device)
    self.goal_dir_b = torch.zeros((self.num_envs, 2), device=self.device)
    self.goal_dir_b[:, 0] = 1.0
    self.goal_obs_age = torch.zeros((self.num_envs, 1), device=self.device)
    self._time_since_update = torch.zeros(self.num_envs, device=self.device)
    self.last_update_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self._latest_stats = ReactiveSoccerOdometryStats(
      localization_success=torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
      visible_landmarks=torch.zeros(self.num_envs, device=self.device),
      pose_error_xy=torch.zeros(self.num_envs, device=self.device),
      pose_error_yaw=torch.zeros(self.num_envs, device=self.device),
      particle_effective_n=None,
    )

    self.proprio_history = CircularBuffer(
      max_len=self.history_steps,
      batch_size=self.num_envs,
      device=self.device,
    )
    self.odometry_model_path = str(odometry_model_path or "")
    self.odometry_model: ReactiveSoccerOdometryMLP | None = None
    self.odometry_model_metadata: ReactiveSoccerOdometryModelMetadata | None = None
    self.odometry_model_extras: dict | None = None
    self.expected_proprio_dim = int(expected_proprio_dim) if expected_proprio_dim is not None else None

    if self.localization_mode in {"learned_only", "particle_filter"}:
      self.odometry_model, self.odometry_model_metadata, self.odometry_model_extras = (
        self._load_odometry_model(self.odometry_model_path)
      )
      if self.odometry_model_metadata.history_steps != self.history_steps:
        raise ValueError(
          "Configured odometry_history_steps does not match checkpoint metadata: "
          f"cfg={self.history_steps}, checkpoint={self.odometry_model_metadata.history_steps}."
        )
      if (
        self.expected_proprio_dim is not None
        and self.odometry_model_metadata.proprio_dim != self.expected_proprio_dim
      ):
        raise ValueError(
          "Configured odometry proprio dim does not match checkpoint metadata: "
          f"cfg={self.expected_proprio_dim}, checkpoint={self.odometry_model_metadata.proprio_dim}."
        )

    self.localizer: ParticleFilterLocalizer | None = None
    if self.localization_mode == "particle_filter":
      self.localizer = ParticleFilterLocalizer(
        num_envs=self.num_envs,
        field_landmarks_f=self.field_landmarks_f,
        device=self.device,
        cfg=pf_cfg or ParticleFilterConfig(),
      )

    self.reset()

  def _load_odometry_model(
    self,
    path: str,
  ) -> tuple[ReactiveSoccerOdometryMLP, ReactiveSoccerOdometryModelMetadata, dict]:
    if not path:
      raise FileNotFoundError(
        "perception.odometry_model_path is required when localization_mode uses learned odometry."
      )
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Odometry checkpoint not found: {checkpoint_path}")
    model, metadata, extras = load_reactive_soccer_odometry_checkpoint(
      checkpoint_path,
      device=str(self.device),
    )
    model.eval()
    return model, metadata, extras

  @property
  def latest_stats(self) -> ReactiveSoccerOdometryStats:
    return self._latest_stats

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    if len(env_ids) == 0:
      return

    gt_pose_f = get_ground_truth_field_pose(self.env)
    self.pose_hat_f[env_ids] = gt_pose_f[env_ids]
    self.prev_delta_pose[env_ids] = 0.0
    self.goal_obs_age[env_ids] = 0.0
    self._time_since_update[env_ids] = 0.0
    self.last_update_mask[env_ids] = False
    if self.proprio_history.is_initialized:
      self.proprio_history.reset(batch_ids=env_ids)
    if self.localizer is not None:
      self.localizer.reset(gt_pose_f, env_ids=env_ids)
    self._refresh_goal_cache(env_ids=env_ids)
    self._refresh_stats(true_pose_f=gt_pose_f, localization_success=None, landmark_observation=None)

  def advance(
    self,
    *,
    projected_gravity: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    previous_action: torch.Tensor,
    true_pose_f: torch.Tensor,
    landmark_observation: LandmarkObservation | None = None,
    landmark_measurement_fresh: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proprio_vec = build_odometry_proprio_vector(
      projected_gravity=projected_gravity,
      base_ang_vel=base_ang_vel,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      previous_action=previous_action,
    )
    if self.expected_proprio_dim is None:
      self.expected_proprio_dim = int(proprio_vec.shape[-1])
    elif proprio_vec.shape[-1] != self.expected_proprio_dim:
      raise ValueError(
        f"Unexpected proprio dim {proprio_vec.shape[-1]}, expected {self.expected_proprio_dim}."
      )
    self.proprio_history.append(proprio_vec)
    self.goal_obs_age += self.step_dt
    self._time_since_update += self.step_dt

    update_ids = (self._time_since_update >= (self.update_period_s - 1.0e-6)).nonzero(
      as_tuple=False
    ).squeeze(-1)
    self.last_update_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    if len(update_ids) > 0:
      self.last_update_mask[update_ids] = True
    localization_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    if len(update_ids) > 0:
      if self.localization_mode == "ground_truth":
        gt_pose = true_pose_f[update_ids]
        delta = compute_body_frame_delta(self.pose_hat_f[update_ids], gt_pose)
        self.pose_hat_f[update_ids] = gt_pose
        self.prev_delta_pose[update_ids] = delta
        self.goal_obs_age[update_ids] = 0.0
        localization_success[update_ids] = True
      else:
        history = self.proprio_history.buffer[update_ids]
        assert self.odometry_model is not None
        with torch.no_grad():
          predicted_delta = self.odometry_model(
            history,
            self.prev_delta_pose[update_ids],
          )
        predicted_delta = torch.nan_to_num(predicted_delta, nan=0.0, posinf=0.0, neginf=0.0)
        predicted_delta[:, 2] = wrap_to_pi(predicted_delta[:, 2])
        self.pose_hat_f[update_ids] = integrate_delta_pose(
          self.pose_hat_f[update_ids],
          predicted_delta,
        )
        self.prev_delta_pose[update_ids] = predicted_delta
        if self.localizer is not None:
          self.localizer.predict(update_ids, predicted_delta)
          if landmark_observation is not None and landmark_measurement_fresh is not None:
            fresh_ids = update_ids[landmark_measurement_fresh[update_ids].bool()]
            if len(fresh_ids) > 0:
              localization_success = self.localizer.update(fresh_ids, landmark_observation)
              success_ids = localization_success.nonzero(as_tuple=False).squeeze(-1)
              if len(success_ids) > 0:
                self.goal_obs_age[success_ids] = 0.0
          self.pose_hat_f[update_ids] = self.localizer.estimate()[update_ids]
        self.pose_hat_f[:, 2] = wrap_to_pi(self.pose_hat_f[:, 2])

      self._time_since_update[update_ids] = (
        self._time_since_update[update_ids] - self.update_period_s
      ).clamp(min=0.0)
      self._refresh_goal_cache(env_ids=update_ids)

    self._refresh_stats(
      true_pose_f=true_pose_f,
      localization_success=localization_success,
      landmark_observation=landmark_observation,
    )
    return self.read()

  def _refresh_goal_cache(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    goal_pos_b, goal_dir_b = goal_observation_from_pose(
      self.pose_hat_f[env_ids],
      self.goal_pos_f,
      self.goal_direction_f,
    )
    self.goal_pos_b[env_ids] = goal_pos_b
    self.goal_dir_b[env_ids] = goal_dir_b

  def _refresh_stats(
    self,
    *,
    true_pose_f: torch.Tensor,
    localization_success: torch.Tensor | None,
    landmark_observation: LandmarkObservation | None,
  ) -> None:
    pose_error_xy = torch.norm(self.pose_hat_f[:, :2] - true_pose_f[:, :2], dim=-1)
    pose_error_yaw = torch.abs(wrap_to_pi(self.pose_hat_f[:, 2] - true_pose_f[:, 2]))
    visible_landmarks = (
      landmark_observation.visible_counts().to(dtype=torch.float32)
      if landmark_observation is not None
      else torch.zeros(self.num_envs, device=self.device)
    )
    self._latest_stats = ReactiveSoccerOdometryStats(
      localization_success=(
        localization_success
        if localization_success is not None
        else torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
      ),
      visible_landmarks=visible_landmarks,
      pose_error_xy=pose_error_xy,
      pose_error_yaw=pose_error_yaw,
      particle_effective_n=(self.localizer.effective_sample_size() if self.localizer is not None else None),
    )

  def read(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      self.goal_pos_b.clone(),
      self.goal_dir_b.clone(),
      self.goal_obs_age.clone(),
    )

  def odometry_artifact_info(self) -> dict[str, object]:
    metadata = (
      self.odometry_model_metadata.to_dict() if self.odometry_model_metadata is not None else None
    )
    return {
      "resolved_path": self.odometry_model_path,
      "localization_mode": self.localization_mode,
      "metadata": metadata,
      "extras": self.odometry_model_extras or {},
    }


ReactiveSoccerOdometryProxy = ReactiveSoccerLearnedOdometryRuntime
