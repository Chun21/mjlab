"""Soccer task observation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  pos_b, _ = subtract_frame_transforms(
    robot.data.root_link_pos_w,
    robot.data.root_link_quat_w,
    ball.data.root_link_pos_w,
    ball.data.root_link_quat_w,
  )
  return pos_b


def ball_vel_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ball: Entity = env.scene["ball"]
  rel_vel_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
  return quat_apply_inverse(robot.data.root_link_quat_w, rel_vel_w)


def goal_pos_b(
  env: ManagerBasedRlEnv,
  goal_pos_w: tuple[float, float, float],
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  goal = torch.tensor(goal_pos_w, device=env.device, dtype=torch.float32).repeat(
    env.num_envs, 1
  )
  pos_b, _ = subtract_frame_transforms(
    robot.data.root_link_pos_w,
    robot.data.root_link_quat_w,
    goal,
  )
  return pos_b


def ball_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  ball: Entity = env.scene["ball"]
  return ball.data.root_link_pos_w


def ball_vel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  ball: Entity = env.scene["ball"]
  return ball.data.root_link_lin_vel_w
