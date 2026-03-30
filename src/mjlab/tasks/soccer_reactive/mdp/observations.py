"""Observation helpers for the reactive soccer task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.soccer.field_specs import M_FIELD
from mjlab.tasks.soccer.mdp import observations as soccer_obs
from mjlab.tasks.soccer_reactive.mdp.perception import detection_probability

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_RIGHT_GOAL_POS_W = (M_FIELD.field_length * 0.5, 0.0, M_FIELD.ball_radius)


def _select_env_ids(obs: torch.Tensor, env_ids=None) -> torch.Tensor:
  if env_ids is None:
    return obs
  return obs[env_ids]


def perceived_ball_pos_b(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  """Return perceived ball position in the robot body frame."""
  ball_pos_b = soccer_obs.ball_pos_b(env)
  mask = perceived_ball_mask(env)
  return _select_env_ids(ball_pos_b * mask, env_ids)


def perceived_ball_mask(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  """Return a deterministic visibility mask based on heading and range."""
  ball_pos_b = soccer_obs.ball_pos_b(env)
  distance = torch.norm(ball_pos_b[:, :2], dim=1)
  in_front = ball_pos_b[:, 0] > 0.0
  prob = detection_probability(distance, near_probability=1.0, falloff=0.15)
  visible = in_front & (prob > 0.1)
  mask = visible.float().unsqueeze(-1)
  return _select_env_ids(mask, env_ids)


def perceived_goal_pos_b(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  """Return perceived target-goal position in the robot body frame."""
  return _select_env_ids(soccer_obs.goal_pos_b(env, _RIGHT_GOAL_POS_W), env_ids)


def perceived_goal_dir_b(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  """Return normalized XY direction to the target goal in the body frame."""
  goal_pos_b = perceived_goal_pos_b(env)
  goal_dir_b = goal_pos_b[:, :2]
  goal_dir_b = goal_dir_b / torch.norm(goal_dir_b, dim=-1, keepdim=True).clamp(
    min=1.0e-6
  )
  return _select_env_ids(goal_dir_b, env_ids)


def privileged_base_lin_vel(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  """Return privileged base linear velocity."""
  return _select_env_ids(envs_mdp.base_lin_vel(env), env_ids)


def privileged_base_height(env: ManagerBasedRlEnv, env_ids=None) -> torch.Tensor:
  """Return privileged base height as a single-column tensor."""
  robot: Entity = env.scene["robot"]
  base_height = robot.data.root_link_pos_w[:, 2:3]
  return _select_env_ids(base_height, env_ids)
