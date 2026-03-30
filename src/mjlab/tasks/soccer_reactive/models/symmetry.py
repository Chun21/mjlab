"""Minimal symmetry helpers for reactive soccer."""

from __future__ import annotations

import torch


class SoccerSymmetry:
  """Mirror action vectors by swapping left/right pairs and flipping signs."""

  def __init__(
    self,
    action_dim: int,
    left_right_pairs: tuple[tuple[int, int], ...],
    sign_flip_indices: tuple[int, ...] = (),
  ) -> None:
    self.action_dim = action_dim
    self.left_right_pairs = tuple(left_right_pairs)
    self.sign_flip_indices = tuple(sign_flip_indices)

  def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
    mirrored = action.clone()
    for left_idx, right_idx in self.left_right_pairs:
      mirrored[..., left_idx] = action[..., right_idx]
      mirrored[..., right_idx] = action[..., left_idx]
    for idx in self.sign_flip_indices:
      mirrored[..., idx] = -mirrored[..., idx]
    return mirrored
