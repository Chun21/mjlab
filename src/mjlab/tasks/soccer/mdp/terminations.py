"""Soccer task termination helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_out_of_bounds(
  env: ManagerBasedRlEnv,
  field_length: float,
  field_width: float,
  margin: float = 0.0,
) -> torch.Tensor:
  ball: Entity = env.scene["ball"]
  x = ball.data.root_link_pos_w[:, 0]
  y = ball.data.root_link_pos_w[:, 1]
  return (torch.abs(x) > field_length * 0.5 + margin) | (
    torch.abs(y) > field_width * 0.5 + margin
  )
