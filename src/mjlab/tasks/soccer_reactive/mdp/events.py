"""Event helpers for the reactive soccer task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.tasks.soccer.field_specs import M_FIELD
from mjlab.utils.lab_api.math import sample_uniform


def reset_ball_only(
  env,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]] | None = None,
) -> None:
  """Reset only the ball to a deterministic in-bounds position."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  ball: Entity = env.scene["ball"]
  env_origins = env.scene.env_origins[env_ids]
  if pose_range is None:
    current_pos = ball.data.root_link_pos_w[env_ids]
    target_x = torch.where(
      torch.abs(current_pos[:, 0] - env_origins[:, 0]) < 0.1,
      env_origins[:, 0] + 0.75,
      env_origins[:, 0],
    )
    target_y = env_origins[:, 1]
    target_z = torch.full_like(target_x, M_FIELD.ball_radius)
    target_pos = torch.stack((target_x, target_y, target_z), dim=-1)
  else:
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device, dtype=torch.float32)
    samples = sample_uniform(
      ranges[:, 0],
      ranges[:, 1],
      (len(env_ids), 3),
      device=env.device,
    )
    target_pos = env_origins[:, :3] + samples
    target_pos[:, 2] += M_FIELD.ball_radius

  default_root_state = ball.data.default_root_state
  assert default_root_state is not None
  pose = default_root_state[env_ids, :7].clone()
  pose[:, :3] = target_pos
  velocity = torch.zeros((len(env_ids), 6), device=env.device, dtype=pose.dtype)

  ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
