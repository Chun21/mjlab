"""Symmetry helpers for reactive soccer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mjlab.asset_zoo.robots import G1_COMP_BODY_JOINT_NAMES


def _build_left_right_pairs(names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
  pairs: list[tuple[int, int]] = []
  index_map = {name: idx for idx, name in enumerate(names)}
  for idx, name in enumerate(names):
    if not name.startswith("left_"):
      continue
    right_name = name.replace("left_", "right_", 1)
    if right_name in index_map:
      pairs.append((idx, index_map[right_name]))
  return tuple(pairs)


def _build_sign_flip_indices(names: tuple[str, ...]) -> tuple[int, ...]:
  sign_flip_indices = [
    idx
    for idx, name in enumerate(names)
    if any(token in name for token in ("roll", "yaw"))
  ]
  return tuple(sign_flip_indices)


@dataclass(frozen=True)
class _ObservationLayout:
  projected_gravity: slice
  base_ang_vel: slice
  joint_pos: slice
  joint_vel: slice
  actions: slice
  ball_pos: slice
  ball_mask: int
  goal_pos: slice
  goal_dir: slice
  ball_obs_age: int
  goal_obs_age: int


class SoccerSymmetry:
  """Mirror observation/action tensors and compute symmetry loss."""

  def __init__(
    self,
    action_dim: int,
    left_right_pairs: tuple[tuple[int, int], ...],
    sign_flip_indices: tuple[int, ...] = (),
    observation_layout: _ObservationLayout | None = None,
  ) -> None:
    self.action_dim = action_dim
    self.left_right_pairs = tuple(left_right_pairs)
    self.sign_flip_indices = tuple(sign_flip_indices)
    self.observation_layout = observation_layout

  @classmethod
  def for_g1_reactive_soccer(cls) -> "SoccerSymmetry":
    joint_pairs = _build_left_right_pairs(tuple(G1_COMP_BODY_JOINT_NAMES))
    sign_flip = _build_sign_flip_indices(tuple(G1_COMP_BODY_JOINT_NAMES))
    observation_layout = _ObservationLayout(
      projected_gravity=slice(0, 3),
      base_ang_vel=slice(3, 6),
      joint_pos=slice(6, 29),
      joint_vel=slice(29, 52),
      actions=slice(52, 75),
      ball_pos=slice(75, 77),
      ball_mask=77,
      goal_pos=slice(78, 80),
      goal_dir=slice(80, 82),
      ball_obs_age=82,
      goal_obs_age=83,
    )
    return cls(
      action_dim=23,
      left_right_pairs=joint_pairs,
      sign_flip_indices=sign_flip,
      observation_layout=observation_layout,
    )

  def _mirror_segment(self, tensor: torch.Tensor) -> torch.Tensor:
    mirrored = tensor.clone()
    for left_idx, right_idx in self.left_right_pairs:
      mirrored[..., left_idx] = tensor[..., right_idx]
      mirrored[..., right_idx] = tensor[..., left_idx]
    for idx in self.sign_flip_indices:
      mirrored[..., idx] = -mirrored[..., idx]
    return mirrored

  def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
    return self._mirror_segment(action)

  def mirror_actor_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
    if self.observation_layout is None:
      raise ValueError("observation_layout is required to mirror actor observations")
    mirrored = actor_obs.clone()
    layout = self.observation_layout

    mirrored[..., layout.projected_gravity.start + 1] = -mirrored[
      ..., layout.projected_gravity.start + 1
    ]
    mirrored[..., layout.base_ang_vel.start + 0] = -mirrored[
      ..., layout.base_ang_vel.start + 0
    ]
    mirrored[..., layout.base_ang_vel.start + 2] = -mirrored[
      ..., layout.base_ang_vel.start + 2
    ]
    mirrored[..., layout.joint_pos] = self._mirror_segment(mirrored[..., layout.joint_pos])
    mirrored[..., layout.joint_vel] = self._mirror_segment(mirrored[..., layout.joint_vel])
    mirrored[..., layout.actions] = self._mirror_segment(mirrored[..., layout.actions])
    mirrored[..., layout.ball_pos.start + 1] = -mirrored[..., layout.ball_pos.start + 1]
    mirrored[..., layout.goal_pos.start + 1] = -mirrored[..., layout.goal_pos.start + 1]
    mirrored[..., layout.goal_dir.start + 1] = -mirrored[..., layout.goal_dir.start + 1]
    return mirrored

  def mirror_actor_history(self, actor_history: torch.Tensor) -> torch.Tensor:
    batch_size, history_steps, obs_dim = actor_history.shape
    mirrored = self.mirror_actor_obs(actor_history.reshape(-1, obs_dim))
    return mirrored.reshape(batch_size, history_steps, obs_dim)

  def symmetry_loss(
    self,
    action: torch.Tensor,
    mirrored_policy_action: torch.Tensor,
  ) -> torch.Tensor:
    return F.mse_loss(action, self.mirror_action(mirrored_policy_action))
