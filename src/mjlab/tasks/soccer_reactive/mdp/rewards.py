"""Minimal reward helpers for the reactive soccer task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.soccer.field_specs import M_FIELD
from mjlab.tasks.soccer.mdp import rewards as soccer_rewards

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_RIGHT_GOAL_POS_W = (M_FIELD.field_length * 0.5, 0.0, M_FIELD.ball_radius)


def _resolve_body_ids(env: "ManagerBasedRlEnv", body_names: tuple[str, ...]) -> tuple[int, ...]:
  cache_name = "_soccer_reactive_body_id_cache"
  cache = getattr(env, cache_name, None)
  if cache is None:
    cache = {}
    setattr(env, cache_name, cache)
  if body_names in cache:
    return cache[body_names]
  body_ids = []
  for body_id in env.scene["robot"].indexing.body_ids.tolist():
    body_name = env.sim.mj_model.body(body_id).name.split("/")[-1]
    if body_name in body_names:
      body_ids.append(body_id)
  if len(body_ids) != len(body_names):
    raise ValueError(f"Failed to resolve body names {body_names}")
  cache[body_names] = tuple(body_ids)
  return cache[body_names]


def goal_scored_reward(
  env: ManagerBasedRlEnv,
  goal_x: float = M_FIELD.field_length * 0.5,
  goal_width: float = M_FIELD.goal_width,
) -> torch.Tensor:
  """Reward successful shots that enter the goal mouth."""
  ball: Entity = env.scene["ball"]
  x = ball.data.root_link_pos_w[:, 0]
  y = ball.data.root_link_pos_w[:, 1]
  return ((x >= goal_x) & (torch.abs(y) <= goal_width * 0.5)).float()


def ball_approach_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Shaping reward for approaching the ball."""
  return soccer_rewards.approach_ball(env)


def robot_ball_alignment_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward aligning the robot heading with the ball direction."""
  return soccer_rewards.face_ball(env)


def goal_progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward ball velocity toward the attacking goal."""
  return soccer_rewards.ball_to_goal_progress(env, _RIGHT_GOAL_POS_W)


def ball_goal_alignment_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward ball velocity pointing toward the goal direction."""
  ball: Entity = env.scene["ball"]
  goal = torch.tensor(_RIGHT_GOAL_POS_W[:2], device=env.device).repeat(env.num_envs, 1)
  goal_dir = goal - ball.data.root_link_pos_w[:, :2]
  goal_dir = goal_dir / torch.norm(goal_dir, dim=1, keepdim=True).clamp(min=1.0e-6)
  vel_xy = ball.data.root_link_lin_vel_w[:, :2]
  speed = torch.norm(vel_xy, dim=1, keepdim=True)
  vel_dir = vel_xy / speed.clamp(min=1.0e-6)
  cosine = torch.sum(vel_dir * goal_dir, dim=1)
  return cosine * speed.squeeze(-1)


def survival_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward staying alive."""
  return envs_mdp.is_alive(env)


def fall_termination_penalty(
  env: ManagerBasedRlEnv,
  termination_term_name: str = "robot_fallen",
) -> torch.Tensor:
  """Penalty that only triggers when the robot falls."""
  return env.termination_manager.get_term(termination_term_name).float()


class StagnationPenalty:
  """Penalize the ball remaining nearly motionless for approximately one second."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._low_speed_steps = torch.zeros(env.num_envs, device=env.device)
    self._step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._low_speed_steps.zero_()
    else:
      self._low_speed_steps[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    threshold: float = 0.05,
    window_s: float = 1.0,
  ) -> torch.Tensor:
    ball: Entity = env.scene["ball"]
    speed = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)
    low_speed = speed < threshold
    self._low_speed_steps[low_speed] += 1.0
    self._low_speed_steps[~low_speed] = 0.0
    return (self._low_speed_steps * self._step_dt >= window_s).float()


def head_pitch_alignment_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Coarse penalty when the ball is vertically misaligned with the base."""
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  height_error = torch.abs(ball.data.root_link_pos_w[:, 2] - robot.data.root_link_pos_w[:, 2])
  return height_error


def head_yaw_alignment_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Coarse heading misalignment penalty based on robot-ball orientation."""
  return 0.5 * (1.0 - robot_ball_alignment_reward(env))


def sideways_kick_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward sideways component of ball motion when the robot is close to the ball."""
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  dist = torch.norm(ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1)
  lateral_speed = torch.abs(ball.data.root_link_lin_vel_w[:, 1])
  return (dist < 0.6).float() * lateral_speed


def forward_kick_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize sending the ball strongly backward in the field frame."""
  backward_speed = (-env.scene["ball"].data.root_link_lin_vel_w[:, 0]).clamp(min=0.0)
  return backward_speed


def touch_ball_reward(
  env: ManagerBasedRlEnv,
  proximity_threshold: float = 0.45,
  speed_scale: float = 1.0,
) -> torch.Tensor:
  """Coarse proxy reward for moving the ball while near it."""
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  dist = torch.norm(ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1)
  speed = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)
  return (dist < proximity_threshold).float() * (speed / max(speed_scale, 1.0e-6))


class touch_ball_impulse_reward:
  """Reward positive increases in ball speed when the robot is near the ball."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.device = env.device
    self._prev_ball_speed = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._prev_ball_speed.zero_()
    else:
      self._prev_ball_speed[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    proximity_threshold: float = 0.45,
    speed_gain_scale: float = 0.25,
  ) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    dist = torch.norm(
      ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2],
      dim=1,
    )
    speed = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)
    speed_gain = (speed - self._prev_ball_speed).clamp(min=0.0)
    self._prev_ball_speed.copy_(speed.detach())
    alignment = robot_ball_alignment_reward(env).clamp(min=0.0)
    return (
      (dist < proximity_threshold).float()
      * alignment
      * (speed_gain / max(speed_gain_scale, 1.0e-6))
    )


def foot_proximity_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the feet getting too close to each other."""
  robot: Entity = env.scene["robot"]
  left_id, right_id = _resolve_body_ids(
    env,
    ("left_ankle_roll_link", "right_ankle_roll_link"),
  )
  feet_pos = robot.data.body_com_pos_w[:, [left_id, right_id], :2]
  feet_dist = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
  return torch.exp(-feet_dist)


def leg_action_rate_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalty on action rate as a proxy for leg action smoothness."""
  return envs_mdp.action_rate_l2(env)


def head_action_rate_penalty(
  env: ManagerBasedRlEnv,
  head_action_indices: tuple[int, ...] = (),
) -> torch.Tensor:
  """Penalty on head-action changes; returns zero when the robot has no head joints."""
  if len(head_action_indices) == 0:
    return torch.zeros(env.num_envs, device=env.device)
  index_tensor = torch.tensor(
    head_action_indices,
    device=env.device,
    dtype=torch.long,
  )
  action_delta = env.action_manager.action[:, index_tensor] - env.action_manager.prev_action[
    :, index_tensor
  ]
  return torch.sum(torch.square(action_delta), dim=1)


def joint_limit_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalty for violating soft joint limits."""
  return envs_mdp.joint_pos_limits(env, asset_cfg=_ROBOT_CFG)


class BaseAccelerationPenalty:
  """Penalize excessive base acceleration using finite differences."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._prev_lin_vel = torch.zeros((env.num_envs, 3), device=env.device)
    self._step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._prev_lin_vel.zero_()
    else:
      self._prev_lin_vel[env_ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    current_lin_vel = robot.data.root_link_lin_vel_w
    acc = (current_lin_vel - self._prev_lin_vel) / max(self._step_dt, 1.0e-6)
    self._prev_lin_vel.copy_(current_lin_vel)
    return torch.sum(torch.square(acc), dim=1)


def non_foot_collision_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str = "non_foot_ground_collision",
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize non-foot collisions with the terrain."""
  sensor = None
  if hasattr(env.scene, "get"):
    sensor = env.scene.get(sensor_name)
  else:
    try:
      sensor = env.scene[sensor_name]
    except Exception:
      sensor = None
  if isinstance(sensor, ContactSensor):
    data = sensor.data
    if data.force_history is not None:
      force_mag = torch.norm(data.force_history, dim=-1)
      hit = (force_mag > force_threshold).any(dim=1)
      return hit.sum(dim=-1).float()
    if data.found is not None:
      return data.found.sum(dim=-1).float()
  fallback = getattr(env, "_soccer_reactive_non_foot_collision_force", None)
  if fallback is not None:
    return fallback.to(env.device)
  return torch.zeros(env.num_envs, device=env.device)


def amp_style_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Read AMP style reward injected by the training runner."""
  reward = getattr(env, "_soccer_reactive_amp_style_reward", None)
  if reward is None:
    return torch.zeros(env.num_envs, device=env.device)
  return reward.to(env.device)


def near_ball_control_reward(
  env: ManagerBasedRlEnv,
  distance_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward smooth actions when the robot is already close to the ball."""
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  dist = torch.norm(ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1)
  action_rate = envs_mdp.action_rate_l2(env)
  return (dist < distance_threshold).float() * torch.exp(-action_rate)
