"""Minimal virtual perception helpers for reactive soccer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def ball_noise_std(distance: torch.Tensor) -> torch.Tensor:
  """Distance-dependent ball observation noise from the paper."""
  return 0.124 * distance + 0.149


def detection_probability(
  distance: torch.Tensor,
  near_probability: float = 1.0,
  falloff: float = 0.0,
) -> torch.Tensor:
  """Simple bounded helper for detection probability."""
  prob = near_probability - falloff * distance
  return torch.clamp(prob, min=0.0, max=1.0)


@dataclass
class VirtualPerceptionState:
  """Sample-and-hold latency buffer for structured perception outputs."""

  latency_steps: int = 0
  update_interval_steps: int = 1
  _obs_history: list[torch.Tensor] = field(default_factory=list, init=False)
  _mask_history: list[torch.Tensor] = field(default_factory=list, init=False)
  _held_obs: torch.Tensor | None = field(default=None, init=False)
  _held_mask: torch.Tensor | None = field(default=None, init=False)
  _push_count: int = field(default=0, init=False)

  def push(self, obs: torch.Tensor, mask: torch.Tensor) -> None:
    """Push a new observation and update held outputs if due."""
    self._obs_history.append(obs.clone())
    self._mask_history.append(mask.clone())
    self._push_count += 1

    history_index = max(0, len(self._obs_history) - self.latency_steps)
    should_update = self._held_obs is None or (
      (self._push_count - 1) % max(self.update_interval_steps, 1) == 0
    )
    if should_update:
      self._held_obs = self._obs_history[history_index]
      self._held_mask = self._mask_history[history_index]

  def read(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the current held observation and detection mask."""
    if self._held_obs is None or self._held_mask is None:
      raise ValueError("VirtualPerceptionState is empty; push observations first.")
    return self._held_obs.clone(), self._held_mask.clone()

  def reset(self) -> None:
    """Clear buffered state."""
    self._obs_history.clear()
    self._mask_history.clear()
    self._held_obs = None
    self._held_mask = None
    self._push_count = 0
