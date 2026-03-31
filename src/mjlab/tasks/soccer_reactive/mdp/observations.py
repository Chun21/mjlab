"""Observation helpers for the reactive soccer task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.soccer.field_specs import M_FIELD
from mjlab.tasks.soccer.mdp import observations as soccer_obs
from mjlab.tasks.soccer_reactive.mdp.perception import (
  detection_probability,
  get_reactive_soccer_perception_runtime,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _select_env_ids(obs: torch.Tensor, env_ids=None) -> torch.Tensor:
  if env_ids is None:
    return obs
  return obs[env_ids]


def _supports_shared_runtime(env: ManagerBasedRlEnv) -> bool:
  return hasattr(env, "cfg") and hasattr(env, "common_step_counter")


def perceived_ball_pos_b(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  if not _supports_shared_runtime(env):
    ball_pos_b = soccer_obs.ball_pos_b(env)[:, :2]
    mask = perceived_ball_mask(env)
    return _select_env_ids(ball_pos_b * mask, env_ids)
  runtime = get_reactive_soccer_perception_runtime(env)
  runtime.update(env)
  ball_pos_b, _ball_mask, _ball_age = runtime.read_ball()
  return _select_env_ids(ball_pos_b, env_ids)


def perceived_ball_mask(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  if not _supports_shared_runtime(env):
    ball_pos_b = soccer_obs.ball_pos_b(env)[:, :2]
    distance = torch.norm(ball_pos_b, dim=1)
    in_fov = ball_pos_b[:, 0] > 0.0
    prob = detection_probability(distance, in_fov=in_fov)
    return _select_env_ids((prob > 0.0).float().unsqueeze(-1), env_ids)
  runtime = get_reactive_soccer_perception_runtime(env)
  runtime.update(env)
  _ball_pos_b, ball_mask, _ball_age = runtime.read_ball()
  return _select_env_ids(ball_mask, env_ids)


def ball_observation_age(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  if not _supports_shared_runtime(env):
    return _select_env_ids(torch.zeros((env.num_envs, 1), device=env.device), env_ids)
  runtime = get_reactive_soccer_perception_runtime(env)
  runtime.update(env)
  _ball_pos_b, _ball_mask, ball_age = runtime.read_ball()
  return _select_env_ids(ball_age, env_ids)


def perceived_goal_pos_b(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  if not _supports_shared_runtime(env):
    goal_pos_b = soccer_obs.goal_pos_b(
      env,
      (M_FIELD.field_length * 0.5, 0.0, M_FIELD.ball_radius),
    )[:, :2]
    return _select_env_ids(goal_pos_b, env_ids)
  runtime = get_reactive_soccer_perception_runtime(env)
  runtime.update(env)
  goal_pos_b, _goal_dir_b, _goal_age = runtime.read_goal()
  return _select_env_ids(goal_pos_b, env_ids)


def perceived_goal_dir_b(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  if not _supports_shared_runtime(env):
    goal_pos_b = perceived_goal_pos_b(env)
    goal_dir_b = goal_pos_b / torch.norm(goal_pos_b, dim=-1, keepdim=True).clamp(
      min=1.0e-6
    )
    return _select_env_ids(goal_dir_b, env_ids)
  runtime = get_reactive_soccer_perception_runtime(env)
  runtime.update(env)
  _goal_pos_b, goal_dir_b, _goal_age = runtime.read_goal()
  return _select_env_ids(goal_dir_b, env_ids)


def goal_observation_age(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  if not _supports_shared_runtime(env):
    return _select_env_ids(torch.zeros((env.num_envs, 1), device=env.device), env_ids)
  runtime = get_reactive_soccer_perception_runtime(env)
  runtime.update(env)
  _goal_pos_b, _goal_dir_b, goal_age = runtime.read_goal()
  return _select_env_ids(goal_age, env_ids)


def privileged_base_lin_vel(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  return _select_env_ids(envs_mdp.base_lin_vel(env), env_ids)


def privileged_ball_true_pos_b(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  return _select_env_ids(soccer_obs.ball_pos_b(env)[:, :2], env_ids)


def privileged_base_height(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  base_height = robot.data.root_link_pos_w[:, 2:3]
  return _select_env_ids(base_height, env_ids)


def _get_env_buffer(
  env: ManagerBasedRlEnv,
  attr_name: str,
  shape: tuple[int, int],
  *,
  fallback: torch.Tensor | None = None,
) -> torch.Tensor:
  value = getattr(env, attr_name, None)
  if value is None:
    value = (
      fallback.to(env.device, dtype=torch.float32)
      if fallback is not None
      else torch.zeros(shape, device=env.device, dtype=torch.float32)
    )
    setattr(env, attr_name, value)
  return value


def privileged_base_mass_randomization(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  root_body_id = int(robot.indexing.root_body_id)
  fallback = env.sim.model.body_mass[:, root_body_id : root_body_id + 1]
  value = _get_env_buffer(
    env,
    "_soccer_reactive_base_mass_randomization",
    (env.num_envs, 1),
    fallback=fallback,
  )
  return _select_env_ids(value, env_ids)


def privileged_base_com_randomization(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  root_body_id = int(robot.indexing.root_body_id)
  fallback = env.sim.model.body_ipos[:, root_body_id, :3]
  value = _get_env_buffer(
    env,
    "_soccer_reactive_base_com_randomization",
    (env.num_envs, 3),
    fallback=fallback,
  )
  return _select_env_ids(value, env_ids)


def privileged_ball_vel_w(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  ball: Entity = env.scene["ball"]
  return _select_env_ids(ball.data.root_link_lin_vel_w[:, :2], env_ids)


def privileged_ball_friction_w(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  ball: Entity = env.scene["ball"]
  ball_geom_id = int(ball.indexing.geom_ids[0])
  fallback = env.sim.model.geom_friction[:, ball_geom_id, :2]
  value = _get_env_buffer(
    env,
    "_soccer_reactive_ball_friction_w",
    (env.num_envs, 2),
    fallback=fallback,
  )
  return _select_env_ids(value, env_ids)


class _PerceptionObservationTerm(ManagerTermBase):
  """Observation term wrapper that resets the shared perception runtime."""

  def __init__(self, cfg, env) -> None:
    del cfg
    super().__init__(env)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if isinstance(env_ids, slice):
      env_ids = None
    runtime = get_reactive_soccer_perception_runtime(self._env)
    runtime.reset(env_ids)


class PerceivedBallPosObs(_PerceptionObservationTerm):
  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return perceived_ball_pos_b(env)


class PerceivedBallMaskObs(_PerceptionObservationTerm):
  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return perceived_ball_mask(env)


class BallObservationAgeObs(_PerceptionObservationTerm):
  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return ball_observation_age(env)


class PerceivedGoalPosObs(_PerceptionObservationTerm):
  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return perceived_goal_pos_b(env)


class PerceivedGoalDirObs(_PerceptionObservationTerm):
  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return perceived_goal_dir_b(env)


class GoalObservationAgeObs(_PerceptionObservationTerm):
  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return goal_observation_age(env)
