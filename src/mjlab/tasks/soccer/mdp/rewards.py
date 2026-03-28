"""Soccer task reward helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def upright_reward(env: ManagerBasedRlEnv, std: float = 0.5) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  xy_sq = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_sq / (std**2))


def approach_ball(env: ManagerBasedRlEnv, std: float = 1.0) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  delta = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  dist_sq = torch.sum(torch.square(delta), dim=1)
  return torch.exp(-dist_sq / (std**2))


def face_ball(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  forward = quat_apply(
    robot.data.root_link_quat_w,
    torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1),
  )
  delta = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  norm = torch.norm(delta, dim=1, keepdim=True).clamp(min=1e-6)
  direction = delta / norm
  return torch.sum(forward[:, :2] * direction, dim=1)


def near_ball_bonus(env: ManagerBasedRlEnv, threshold: float = 0.4) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  dist = torch.norm(
    ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2],
    dim=1,
  )
  return (dist < threshold).float()


def ball_to_goal_progress(
  env: ManagerBasedRlEnv,
  goal_pos_w: tuple[float, float, float],
) -> torch.Tensor:
  ball: Entity = env.scene["ball"]
  goal = torch.tensor(goal_pos_w[:2], device=env.device).repeat(env.num_envs, 1)
  goal_dir = goal - ball.data.root_link_pos_w[:, :2]
  goal_dir = goal_dir / torch.norm(goal_dir, dim=1, keepdim=True).clamp(min=1e-6)
  return torch.sum(ball.data.root_link_lin_vel_w[:, :2] * goal_dir, dim=1)
