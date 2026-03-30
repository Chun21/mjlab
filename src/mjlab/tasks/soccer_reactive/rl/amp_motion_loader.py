"""Minimal AMP motion loader for reactive soccer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mjlab.asset_zoo.robots import G1_COMP_BODY_JOINT_NAMES


class ReactiveSoccerAmpMotionLoader:
  """Sample simple transition batches from existing soccer AMP clips."""

  def __init__(self, motion_root: Path, device: str = 'cpu') -> None:
    self.motion_root = Path(motion_root)
    self.device = device
    self._clips = self._load_clips()
    self.transition_dim = int(self._clips[0].shape[1])

  def _load_clips(self) -> list[torch.Tensor]:
    motion_files = sorted(self.motion_root.rglob('*.npz'))
    if not motion_files:
      raise ValueError(f'No AMP motions found under {self.motion_root}')
    clips: list[torch.Tensor] = []
    for motion_file in motion_files:
      data = np.load(motion_file, allow_pickle=False)
      joint_names = [str(name) for name in data['joint_names']]
      joint_indices = [joint_names.index(name) for name in G1_COMP_BODY_JOINT_NAMES]
      joint_pos = torch.tensor(data['joint_pos'][:, joint_indices], device=self.device, dtype=torch.float32)
      joint_vel = torch.tensor(data['joint_vel'][:, joint_indices], device=self.device, dtype=torch.float32)
      transition = torch.cat((joint_pos[:-1], joint_vel[:-1], joint_pos[1:], joint_vel[1:]), dim=-1)
      clips.append(transition)
    return clips

  def sample_transition_batch(self, batch_size: int) -> torch.Tensor:
    if batch_size <= 0:
      raise ValueError('batch_size must be positive')
    samples = []
    clip_ids = torch.randint(len(self._clips), (batch_size,), device=self.device)
    for clip_id in clip_ids.tolist():
      clip = self._clips[clip_id]
      frame_id = int(torch.randint(clip.shape[0], (1,), device=self.device).item())
      samples.append(clip[frame_id])
    return torch.stack(samples, dim=0)
